#!/bin/bash
SAVE_DIR="./datas" 

# download data
if [ ! -d $SAVE_DIR ]; then
    # create dir
    echo "create 1D dir"
    mkdir "$SAVE_DIR"
    mkdir "$SAVE_DIR/data"
    
    # download 1D datasets
    mkdir "$SAVE_DIR/data/1D"

    # Advection
    mkdir "$SAVE_DIR/data/1D/Advection"
    PREFIX="$SAVE_DIR/data/1D/Advection"
    wget "https://darus.uni-stuttgart.de/api/access/datafile/133100" -P "$PREFIX"
    mv "$PREFIX/133100" "$PREFIX/1D_Advection_Sols_beta0.1.hdf5"

    # CFD
    mkdir "$SAVE_DIR/data/1D/CFD"
    PREFIX="$SAVE_DIR/data/1D/CFD"
    wget "https://darus.uni-stuttgart.de/api/access/datafile/135485" -P "$PREFIX"
    mv "$PREFIX/135485" "$PREFIX/1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5"

    # download 2D datasets
    mkdir "$SAVE_DIR/data/2D"

    # Diffusion-Reaction
    mkdir "$SAVE_DIR/data/2D/DiffReact"
    PREFIX="$SAVE_DIR/data/2D/DiffReact"
    wget "https://darus.uni-stuttgart.de/api/access/datafile/133017" -P "$PREFIX"
    mv "$PREFIX/133017" "$PREFIX/2D_diff-react_NA_NA.h5"

    # SWE
    mkdir "$SAVE_DIR/data/2D/SWE"
    PREFIX="$SAVE_DIR/data/2D/SWE"
    wget "https://darus.uni-stuttgart.de/api/access/datafile/133021" -P "$PREFIX"
    mv "$PREFIX/133021" "$PREFIX/2D_rdb_NA_NA.h5"

else
    echo "$SAVE_DIR already exist (please remove before action)"
fi

# Data would be splitted while running
