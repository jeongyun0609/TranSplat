##Category: Trash
gdown --folder https://drive.google.com/drive/folders/1iXZiYpRGqzs1AbKr17W9jQr3smRy7NGv?usp=drive_link
cd Trash_Only
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../train
cd ../..
rm -rf Trash_Only

##Category: Household
gdown --folder https://drive.google.com/drive/folders/1S3J84ecZVR2mzX8ksQfuPzp6aazVQdHo?usp=drive_link
cd Household_only
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../train
cd ../..
rm -rf Household_only

##Category: Chemical
gdown --folder https://drive.google.com/drive/folders/1ssi9-m_pviSwAJ8Z7M7p19vTuB3b9gBW?usp=drive_link
cd Chemical_Only
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../train
cd ../..
rm -rf Chemical_Only

##Category: All
gdown --folder https://drive.google.com/drive/folders/18j_F9j135FOzYYU76UJ2RR6uc0EKFrTd?usp=drive_link
cd All\ Object
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../train
cd ../..
rm -rf All\ Object

#Test seq
gdown --folder https://drive.google.com/drive/folders/1QLEt3jBY8ertAWnZUTh74ROUrVeW_WO6?usp=drive_link
cd Testset
for file in *.tar.gz; do
    tar -xzf $file
done
cd sequences
mv * ../../test
cd ../..
rm -rf Testset