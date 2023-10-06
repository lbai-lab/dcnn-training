#!/bin/bash
SAVE_DIR="./datas" 

# download data
if [ ! -d $SAVE_DIR ]; then
    # create dir
    echo "create $SAVE_DIR dir"
    mkdir $SAVE_DIR

    # 7 of the molecules
    for m in aspirin ethanol malonaldehyde naphthalene salicylic toluene uracil
    do
        wget "http://www.quantum-machine.org/gdml/data/npz/md17_$m.npz" -P $SAVE_DIR
        mv "$SAVE_DIR/md17_$m.npz" "$SAVE_DIR/${m}_dft.npz"
    done

    # last molecule
    wget "http://www.quantum-machine.org/gdml/data/npz/md17_benzene2017.npz" -P $SAVE_DIR
    mv "$SAVE_DIR/md17_benzene2017.npz" "$SAVE_DIR/benzene_dft.npz"
else
    echo "$SAVE_DIR already exist (please remove before action)"
fi

# random split data (default: 80% train, 10% valid, 10% test)
python split_data.py 
