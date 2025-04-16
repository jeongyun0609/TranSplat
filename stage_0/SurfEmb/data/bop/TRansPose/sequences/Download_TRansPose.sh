##Category: Trash
gdown --folder https://drive.google.com/drive/folders/12RE27Jjh2o6g5K4CxBCV8g-s32JXuEKp?usp=drive_link
cd Trash_Only
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../train
cd ../..
rm -rf Trash_Only

##Category: Household
gdown --folder https://drive.google.com/drive/folders/1HxJiFkdSp-kije2Jz5QcSxlS_uDoNyr6?usp=drive_link
cd Household_only
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../train
cd ../..
rm -rf Household_only

##Category: Chemical
gdown --folder https://drive.google.com/drive/folders/1ObRKzQ5drQPs7WNiXzxvWy_J0rpj2HKZ?usp=drive_link
cd Chemical_Only
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../train
cd ../..
rm -rf Chemical_Only

##Category: All
gdown --folder https://drive.google.com/drive/folders/1gTV22EqIOnWCNnuO8-GEEgtjlWs9Darz?usp=drive_link
cd All\ Object
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../train
cd ../..
rm -rf All\ Object

#Test seq
gdown --folder https://drive.google.com/drive/folders/1dcHB3lL-35_6otSGzDqw7ggFeG9U7g5p?usp=drive_link
cd Testset
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../test
cd ../..
rm -rf Testset