import numpy as np
import torch
import cv2
import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Type, List
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import random
import time

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
set_random_seed(3407)


import os
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image

class LabPicsDataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = data
        self.length = len(data)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ent = self.data[idx]
        Img = Image.open(ent["image"]).convert('RGB')
        if self.transform:
            Img = self.transform(Img)
        mask = np.zeros((1, 256, 256), dtype=np.uint8)
        mask_path = ent["annotation"]
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask[0, :, :] = cv2.resize(mask_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask[mask != 0 ] = 1
        if Img is None or np.sum(np.array(mask)) == 0:
            return self.__getitem__(random.randint(0, self.length - 1))
        return Img, mask, ent["id"]
transform = transforms.Compose([
    transforms.Resize((1024, 1024)), 
    transforms.ToTensor(),
])

def collate_fn(batch):
    images, masks, ids = [], [], []
    for (Img, mask, id) in batch:
        images.append(np.array(Img))
        masks.append(mask)
        ids.append(id)
    images_out = torch.from_numpy(np.array(images)) 
    masks_out = np.array(masks)
    ids_out = np.array(ids)

    return images_out, masks_out, ids_out

class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim):
        super(MaskDecoder, self).__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = 7
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.transformer = TwoWayTransformer(depth=2, embedding_dim=transformer_dim, num_heads=8, mlp_dim=2048)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            nn.LayerNorm([transformer_dim // 4, 128, 128]),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            nn.GELU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
    def get_mask(self, sparse_embedding, dense_embedding, image_pe, id):
        
        output_tokens = torch.tensor(np.zeros((dense_embedding.size(0), 1, self.transformer_dim)).astype(np.float32)).to('cuda')
        for i, single_id in enumerate(id):
            output_tokens[i] = self.mask_tokens.weight[single_id].unsqueeze(0)
        tokens = torch.cat((output_tokens, sparse_embedding), dim=1)
        src = dense_embedding
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = torch.tensor(np.zeros((dense_embedding.size(0), 1, 32)).astype(np.float32)).to('cuda')
        for i, single_id in enumerate(id):
            hyper_in[i] = self.output_hypernetworks_mlps[single_id](mask_tokens_out[i,0, :].unsqueeze(0).unsqueeze(1))
        b, c, h, w = upscaled_embedding.shape
        mask = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w).squeeze(1)
        return mask, hyper_in, upscaled_embedding
    def forward(self, sparse_embedding, dense_embedding, image_pe, id):
        mask, hyper_in, upscaled_embedding = self.get_mask(sparse_embedding, dense_embedding, image_pe, id)
        return mask

class MaskFormer(nn.Module):
    def __init__(self, transformer_dim):
        super(MaskFormer, self).__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = 7
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.transformer = TwoWayTransformer(depth=2, embedding_dim=transformer_dim, num_heads=8, mlp_dim=2048)
    def get_mask(self, image_embeddings, image_pe, id):
        tokens = torch.tensor(np.zeros((image_embeddings.size(0), 1, self.transformer_dim)).astype(np.float32)).to('cuda')
        for i, single_id in enumerate(id):
            tokens[i] = self.mask_tokens.weight[single_id].unsqueeze(0)
        src = image_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs
        src = src.transpose(1, 2).view(b, c, h, w)
        return mask_tokens_out, src + image_embeddings
    def forward(self, image_embeddings, image_pe, id):
        sparse_feat, dense_feat = self.get_mask(image_embeddings, image_pe, id)
        return sparse_feat, dense_feat

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=nn.ReLU
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

def get_positional_encoding(height, width, d):
    row = torch.arange(height).unsqueeze(1)  # (height, 1)
    col = torch.arange(width).unsqueeze(0)  # (1, width)
    div_term = torch.exp(torch.arange(0, d, 2) * -(torch.log(torch.tensor(10000.0)) / d))  # (d/2,)
    pe = torch.zeros((height, width, d))
    pe[:, :, 0::2] = torch.sin(row * div_term)  # 对应sin
    pe[:, :, 1::2] = torch.cos(col.T * div_term)  # 对应cos

    return pe.permute(2, 0, 1).unsqueeze(0)



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.image_pe = get_positional_encoding(64, 64, 256)
        self.mask_decoder = MaskDecoder(transformer_dim=256)
    def forward(self, sparse_embedding, dense_embedding, id):
        #print(x.shape)
        mask = self.mask_decoder(sparse_embedding, dense_embedding, self.image_pe.to('cuda'), id)
        return mask

class FeatModel(nn.Module):
    def __init__(self):
        super(FeatModel, self).__init__()
        self.image_pe = get_positional_encoding(64, 64, 256)
        self.mask_former = MaskFormer(transformer_dim=256)
    def forward(self, image_embedding, id):
        #print(x.shape)
        sparse_embedding, dense_embedding = self.mask_former(image_embedding, self.image_pe.to('cuda'), id)
        return sparse_embedding, dense_embedding

from rep_vit import rep_vit

if __name__=="__main__":
    
    rep_vit_model = rep_vit.rep_vit_m1(1024)
    rep_model_path = './epoch_35.pth'
    if os.path.exists(rep_model_path):
        rep_vit_model.load_state_dict(torch.load(rep_model_path))
    pic_size = -1

    class_compensate = [1, 1, 1, 1, 1, 1, 1]
    data_dir=r"./your_data_dir" 
    data=[]
    sub_folder_list = ['BF', 'CA', 'LND', 'MCS', 'PF', 'SI', 'UP']
    for id, sub_folder in enumerate(sub_folder_list):
        for ff, name in enumerate(os.listdir(data_dir + "train_gt/" + sub_folder + '/')):  # go over all folder annotation
            if ff >= pic_size and pic_size != -1:
                break
            new_name = name.split('.')[0] + ".png"
            for i in range(class_compensate[id]):
                data.append({"image":data_dir+"train_og/"+new_name,"annotation":data_dir + "train_gt/" + sub_folder + '/'+name,"id":id})
    model = Model()
    feat_model = FeatModel()
    rep_vit_model = rep_vit_model.to('cuda')
    model = model.to('cuda')
    feat_model = feat_model.to('cuda')
    rep_vit_model.train()
    model.train()
    feat_model.train()
    old_model_path = "./model.torch"
    old_feat_model_path = "./feat_model.torch"
    if os.path.exists(old_model_path):
        model.load_state_dict(torch.load(old_model_path))
        feat_model.load_state_dict(torch.load(old_feat_model_path))
        print("Load", old_model_path)
    optimizer = torch.optim.AdamW(
        params=list(rep_vit_model.parameters()) + list(feat_model.parameters()) + list(model.parameters()),
        lr=5e-4,  
        weight_decay=1e-5
    )
    def lr_lambda(epoch):
        if epoch < 15:      
            return 1.0
        else:              
            return 0.2      
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )


    dataset = LabPicsDataset(data, transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=8)

    # Training loop

    log_file_path = './sam_endo18_rep_log.txt'
    save_folder = './sam_endo18_rep'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    def mean(x):
        if len(x) == 0: return 0
        return sum(x)/len(x)
    for itr in range(0, 31):
        iou_list =[]
        for i in range(7): iou_list.append([])
        eps=1e-6
        for batch_idx, batch in enumerate(dataloader):
            image,mask,id, = batch # load data batch
            image = image.to('cuda')
            id = torch.from_numpy(np.array(id)).to(torch.int16).to('cuda')
            start_time = time.perf_counter_ns()
            vit_feat = rep_vit_model(image)
            sparse_feat, dense_feat = feat_model(vit_feat, id)
            prd_masks = model(sparse_feat, dense_feat, id)
            prd_mask = torch.sigmoid(prd_masks)
            gt_mask = torch.tensor(mask[:, 0].astype(np.float32)).cuda()
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss
            loss = seg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.perf_counter_ns()
            print('time cost', end_time - start_time)
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = (inter + eps) / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter + eps)
            
            
            iou = (iou.cpu().detach().numpy())
            for i, single_id in enumerate(id):
                iou_list[single_id].append(iou[i])
            mean_iou =  1.0 * (sum(iou_list[0]) + sum(iou_list[1]) + sum(iou_list[2]) + sum(iou_list[3]) + sum(iou_list[4]) + sum(iou_list[5]) + sum(iou_list[6])) / (len(iou_list[0]) + len(iou_list[1]) + len(iou_list[2]) + len(iou_list[3]) + len(iou_list[4]) + len(iou_list[5]) + len(iou_list[6]))
            print("step)", itr, "batch:", batch_idx+1, '/', len(dataloader), "Accuracy(IOU)=", mean_iou, "Loss=", loss.cpu().detach().numpy().mean(), "LR=", optimizer.param_groups[0]['lr'])
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{mean(iou_list[0]):.8f} {mean(iou_list[1]):.8f} {mean(iou_list[2]):.8f} {mean(iou_list[3]):.8f} {mean(iou_list[4]):.8f} {mean(iou_list[5]):.8f} {mean(iou_list[6]):.8f}\n")
                log_file.write(f"itr: {itr}, batch: {batch_idx+1}/{len(dataloader)}, iou: {mean_iou:.8f}, loss: {loss.cpu().detach().numpy().mean():.8f}\n")

        print("step)", itr, "All Accuracy(IOU)=", np.array(iou_list[0]).mean(), np.array(iou_list[1]).mean(), np.array(iou_list[2]).mean(), np.array(iou_list[3]).mean(), np.array(iou_list[4]).mean(), np.array(iou_list[5]).mean(), np.array(iou_list[6]).mean())
        scheduler.step()
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{np.array(iou_list[0]).mean():.8f}\n")
        torch.save(model.state_dict(), '{1}/sam_endo18_rep_{0}.torch'.format(itr, save_folder)) # save model
        torch.save(feat_model.state_dict(), '{1}/sam_endo18_rep_feat_{0}.torch'.format(itr, save_folder)) # save model
        torch.save(rep_vit_model.state_dict(), '{1}/sam_endo18_rep_rep_{0}.torch'.format(itr, save_folder)) # save model