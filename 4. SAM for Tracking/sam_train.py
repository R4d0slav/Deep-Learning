#!/usr/bin/python3

from sam_utils import *


trainset = SegmentationDataset(train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False, num_workers=2)

testset = SegmentationDataset(train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)


prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size

sam = Sam(
  image_encoder=ImageEncoderViT(
      depth=32,
      embed_dim=1280,
      img_size=image_size,
      mlp_ratio=4,
      norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
      num_heads=16,
      patch_size=vit_patch_size,
      qkv_bias=True,
      use_rel_pos=True,
      global_attn_indexes=[7, 15, 23, 31],
      window_size=14,
      out_chans=prompt_embed_dim,
  ),
  prompt_encoder=PromptEncoder(
      embed_dim=prompt_embed_dim,
      image_embedding_size=(image_embedding_size, image_embedding_size),
      input_image_size=(image_size, image_size),
      mask_in_chans=16,
  ),
  mask_decoder=MaskDecoder(
      num_multimask_outputs=3,
      transformer=TwoWayTransformer(
          depth=2,
          embedding_dim=prompt_embed_dim,
          mlp_dim=2048,
          num_heads=8,
      ),
      transformer_dim=prompt_embed_dim,
      iou_head_depth=3,
      iou_head_hidden_dim=256,
  ),
  pixel_mean=[123.675, 116.28, 103.53],
  pixel_std=[58.395, 57.12, 57.375],
)
sam.train()

with open("sam_vit_h_4b8939.pth", "rb") as f:
    state_dict = torch.load(f)
sam.load_state_dict(state_dict)

sam.to(device='cuda')
print("Model initialized!!")

transform = ResizeLongestSide(sam.image_encoder.img_size)

sam.mask_decoder.my_conv = nn.Conv2d(2, 256, kernel_size=1)
sam.mask_decoder.my_sparse_embeddings = nn.Embedding(1, 256)
sam.mask_decoder.my_attention = nn.MultiheadAttention(256, 8, 0.1, batch_first=True)

@torch.enable_grad()
def train_model(mask_decoder):

    N = 1000
    epochs = 5

    # for n, p in mask_decoder.named_parameters():
    #     print(n, p.requires_grad)

    criterion = nn.BCELoss()
    mask_decoder.train().cuda()

    optimizer = optim.Adam(mask_decoder.parameters(), lr=0.0001)

    for epoch in range(epochs):
        with tqdm.tqdm(total=N, desc = str(epoch+1)+"/"+str(epochs), miniters=1, unit='img') as prog_bar:
          for i, data in enumerate(trainloader, 0):
              if i >= N:
                  break

              template = data["template"][0].cuda()
              template_mask = data["template_mask"][0].cuda()
              image = data["image"][0].cuda()
              image_mask = data["mask"][0].cuda()

              # >> ENCODER <<
              # -------------
              template_embeddings, transformed_template = encode(sam.image_encoder, transform, template)
              image_embeddings, transformed_image = encode(sam.image_encoder, transform, image)

              # >> ATTENTION <<
              # ---------------
              resized_template_mask = cv2.resize(template_mask.clone().cpu().numpy()[0], (template_embeddings.shape[2], template_embeddings.shape[2]))
              masksT = torch.stack([torch.from_numpy(resized_template_mask), torch.from_numpy(1-resized_template_mask)]).unsqueeze(0).float()
              output = mask_decoder.my_conv(masksT.cuda().float())

              Q = image_embeddings.flatten(-2).permute(0, 2, 1).cuda()
              K = template_embeddings.flatten(-2).permute(0, 2, 1).cuda()
              V = output.flatten(-2).permute(0, 2, 1).cuda()

              attention_output = mask_decoder.my_attention(Q, K, K+V)[0]
              attention_output = attention_output.permute(0, 2, 1).reshape(1, 256, 64, 64)

              # >> DECODER <<
              # -------------

              low_res_masks, iou_predictions = mask_decoder(
                  image_embeddings = image_embeddings.cuda(),
                  image_pe = sam.prompt_encoder.get_dense_pe().cuda(),
                  sparse_prompt_embeddings = mask_decoder.my_sparse_embeddings.weight.view(1, 1, 256).cuda(),
                  dense_prompt_embeddings = attention_output.cuda(),
                  multimask_output=False,
              )

              original_size = image.shape[1:]
              input_size = tuple(transformed_image.shape)[2:]

              masks = postprocess_masks(
                  low_res_masks.sigmoid(),
                  input_size,
                  original_size,
                  sam.image_encoder
              )
              # plt.imshow(masks.cpu().clone().detach().numpy()[0][0])
              # plt.show()
              # print(low_res_masks.sigmoid().min(), low_res_masks.sigmoid().max())
              # print(mask_decoder.my_sparse_embeddings.weight.sum())
              # outputs = masks > 0.0 # threshold

              # zero the parameter gradients
              optimizer.zero_grad()

              loss = criterion(masks[0][0], image_mask[0])

              loss.backward()

              optimizer.step()

              prog_bar.set_postfix(**{'loss': np.round(loss.data.cpu().detach().numpy(),5)})
              prog_bar.update(1)

        # Save model checkpoint
        if not os.path.exists("trained"):
            os.makedirs("trained")
        net_path = "trained/sam_mask_decoder_e%d.pth" % (epoch + 1)
        torch.save(mask_decoder.state_dict(), net_path)

    return mask_decoder

print("Starting the training..")
sam_mask_decoder = train_model(sam.mask_decoder)
print("Finished!!")
