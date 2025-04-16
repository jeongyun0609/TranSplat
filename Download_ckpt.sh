##SurfEmb stable-diffusion model ckpt
gdown https://drive.google.com/uc?id=1KrMFEAQorhVmfAnldkamsEdmbe13w_8I
mv SurfEmb_sd.ckpt stage_1/models/SurfEmb_sd.ckpt

##SurfEmb TRansPose model ckpt
mkdir stage_0/SurfEmb/data/model
gdown https://drive.google.com/uc?id=1efbYI069_ZGUuXeYEiL-k0a7AH6uSikj
mv TRansPose-SurfEmb.ckpt stage_0/SurfEmb/data/model/TRansPose-SurfEmb.ckpt
