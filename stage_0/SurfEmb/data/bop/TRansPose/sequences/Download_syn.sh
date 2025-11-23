gdown --folder https://drive.google.com/drive/folders/1ddGcskSd9YJ2WRoM0vgNW1O2lbOiHArY?usp=drive_link

cd TRansPose_syn
for file in *.tar.gz; do
    tar -xzf $file
done
rm -rf *.tar.gz
cd ..
mv TRansPose_syn ../../TRansPose_syn

gdown --folder https://drive.google.com/drive/folders/1X5nx2-TavYDvXCMCoE8-CgYS47JFYKYc?usp=drive_link
cd ClearPose_syn
for file in *.tar.gz; do
    tar -xzf $file
done
rm -rf *.tar.gz
cd ..
mv ClearPose_syn ../../ClearPose_syn