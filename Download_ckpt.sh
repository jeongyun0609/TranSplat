##SurfEmb stable-diffusion model ckpt
gdown https://drive.google.com/uc?id=1X0gifajS40D4l2nl8C-98k6gblnfsam3
mv SurfEmb_sd.ckpt stage_1/models/SurfEmb_sd.ckpt

##SurfEmb TRansPose model ckpt
mkdir stage_0/SurfEmb/data/model
gdown https://drive.google.com/uc?id=1I3Ml8GQTTDdIkbndGvUUTzLGVqeY135K
mv TRansPose-SurfEmb.ckpt stage_0/SurfEmb/data/model/TRansPose-SurfEmb.ckpt
