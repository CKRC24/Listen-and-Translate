#/usr/bin bash
wget -O "data/test_input_data.npy" "https://www.dropbox.com/s/wx3ig36glym448b/test_input_data.npy?dl=1"
wget -O "model/stacking_ds_512_1.h5" "https://www.dropbox.com/s/fjr4psjekt42ssq/stacking_ds_512_1.h5?dl=1"
wget -O "model/stacking_ds_512_2.h5" "https://www.dropbox.com/s/h2g4v8xlqcm5a5r/stacking_ds_512_2.h5?dl=1"
wget -O "model/stacking_ds_512_3.h5" "https://www.dropbox.com/s/eoxwsvdu1u3jtxu/stacking_ds_512_3.h5?dl=1"
wget -O "model/stacking_ds_512_4.h5" "https://www.dropbox.com/s/gta391ii40cu021/stacking_ds_512_4.h5?dl=1"
wget -O "model/stacking_ds_512_5.h5" "https://www.dropbox.com/s/uo8y86xibdumaqo/stacking_ds_512_5.h5?dl=1"
wget -O "model/stacking_ds_ver2_512_1.h5" "https://www.dropbox.com/s/txz87kko8knjfj9/stacking_ds_ver2_512_1.h5?dl=1"
wget -O "model/stacking_ds_ver2_512_2.h5" "https://www.dropbox.com/s/roe69vxf81prhxe/stacking_ds_ver2_512_2.h5?dl=1"
wget -O "model/stacking_ds_ver2_512_3.h5" "https://www.dropbox.com/s/okj58ww03an81ta/stacking_ds_ver2_512_3.h5?dl=1"
wget -O "model/stacking_ds_ver2_512_4.h5" "https://www.dropbox.com/s/hza4auul05xyn1m/stacking_ds_ver2_512_4.h5?dl=1"

python3 src/retrieval_predict.py $1
