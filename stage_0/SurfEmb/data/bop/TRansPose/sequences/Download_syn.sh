gdown --folder https://drive.google.com/drive/folders/1zAL-Wfq9uyD6Prw8_XBP9tWpfWynnufq?usp=drive_link
cd TRansPose_syn
for file in *.tar.gz; do
    tar -xzf $file
done
rm -rf *.tar.gz
cd ..
mv TRansPose_syn ../../TRansPose_syn

gdown --folder https://drive.google.com/drive/folders/1fSMFNIGCh2_BG37jo8fYpcFBMMw_rihJ?usp=drive_link
cd ClearPose_syn
for file in *.tar.gz; do
    tar -xzf $file
done
rm -rf *.tar.gz
cd ..
mv ClearPose_syn ../../ClearPose_syn