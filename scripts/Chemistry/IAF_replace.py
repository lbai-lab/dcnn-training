from scripts.IAFs import *

def get_activation(activation, g):
    # get IAF, only relu_int is used for now
    if activation == "int_relu": return relu_int(g)
    if activation == "int_leaky_relu": return leaky_relu_int(g)
    if activation == "int_elu": return elu_int(g)

    raise ValueError("activation name incorrect")

def replace_activation(model, activation, g):
    # replace activation layers in each NN instance, all activation layers are lisdted.
    act = get_activation(activation, g)

    if model.__class__.__name__ == "SchNetWrap":
        model.interactions[0].mlp[1] = act
        model.interactions[0].act = act
        model.interactions[1].mlp[1] = act
        model.interactions[1].act = act
        model.interactions[2].mlp[1] = act
        model.interactions[2].act = act
        model.interactions[3].mlp[1] = act
        model.interactions[3].act = act
        model.interactions[4].mlp[1] = act
        model.interactions[4].act = act
        model.act = act
    elif model.__class__.__name__ ==  "CGCNN" or \
        model.__class__.__name__ == "CGCNN_NO_BN":
        model.conv_to_fc[1] = act
        model.fcs[1] = act
        model.fcs[3] = act
    elif model.__class__.__name__ == "DimeNetPlusPlusWrap":
        #model.emb.act = act
        #model.interaction_blocks[0].act = act
        #model.interaction_blocks[1].act = act
        #model.interaction_blocks[2].act = act
        model.output_blocks[0].act = act
        model.output_blocks[1].act = act
        model.output_blocks[2].act = act
        model.output_blocks[3].act = act
    elif model.__class__.__name__ == "ForceNet" or \
        model.__class__.__name__ == "ForceNet_NO_BN":
        model.interactions[0].mlp_edge[1] = act
        model.interactions[0].mlp_edge[3] = act
        model.interactions[0].mlp_trans[2] = act
        model.interactions[1].mlp_edge[1] = act
        model.interactions[1].mlp_edge[3] = act
        model.interactions[1].mlp_trans[2] = act
        model.interactions[2].mlp_edge[1] = act
        model.interactions[2].mlp_edge[3] = act
        model.interactions[2].mlp_trans[2] = act
        model.interactions[3].mlp_edge[1] = act
        model.interactions[3].mlp_edge[3] = act
        model.interactions[3].mlp_trans[2] = act
        model.interactions[4].mlp_edge[1] = act
        model.interactions[4].mlp_edge[3] = act
        model.interactions[4].mlp_trans[2] = act
        model.activation = act
    elif model.__class__.__name__ == "GemNetT":
        #model.edge_emb.dense._activation = act
        model.out_blocks[0].layers[0]._activation = act
        model.out_blocks[0].layers[1].dense_mlp[0]._activation = act
        model.out_blocks[0].layers[1].dense_mlp[1]._activation = act
        model.out_blocks[0].layers[2].dense_mlp[0]._activation = act
        model.out_blocks[0].layers[2].dense_mlp[1]._activation = act
        model.out_blocks[0].layers[3].dense_mlp[0]._activation = act
        model.out_blocks[0].layers[3].dense_mlp[1]._activation = act
        model.out_blocks[1].layers[0]._activation = act
        model.out_blocks[1].layers[1].dense_mlp[0]._activation = act
        model.out_blocks[1].layers[1].dense_mlp[1]._activation = act
        model.out_blocks[1].layers[2].dense_mlp[0]._activation = act
        model.out_blocks[1].layers[2].dense_mlp[1]._activation = act
        model.out_blocks[1].layers[3].dense_mlp[0]._activation = act
        model.out_blocks[1].layers[3].dense_mlp[1]._activation = act
        model.out_blocks[2].layers[0]._activation = act
        model.out_blocks[2].layers[1].dense_mlp[0]._activation = act
        model.out_blocks[2].layers[1].dense_mlp[1]._activation = act
        model.out_blocks[2].layers[2].dense_mlp[0]._activation = act
        model.out_blocks[2].layers[2].dense_mlp[1]._activation = act
        model.out_blocks[2].layers[3].dense_mlp[0]._activation = act
        model.out_blocks[2].layers[3].dense_mlp[1]._activation = act
        model.out_blocks[3].layers[0]._activation = act
        model.out_blocks[3].layers[1].dense_mlp[0]._activation = act
        model.out_blocks[3].layers[1].dense_mlp[1]._activation = act
        model.out_blocks[3].layers[2].dense_mlp[0]._activation = act
        model.out_blocks[3].layers[2].dense_mlp[1]._activation = act
        model.out_blocks[3].layers[3].dense_mlp[0]._activation = act
        model.out_blocks[3].layers[3].dense_mlp[1]._activation = act
        # model.int_blocks[0].dense_ca._activation = act
        # model.int_blocks[0].trip_interaction.dense_ba._activation = act
        # model.int_blocks[0].trip_interaction.down_projection._activation = act
        # model.int_blocks[0].trip_interaction.up_projection_ca._activation = act
        # model.int_blocks[0].trip_interaction.up_projection_ac._activation = act
        # model.int_blocks[0].layers_before_skip[0].dense_mlp[0]._activation = act
        # model.int_blocks[0].layers_before_skip[0].dense_mlp[1]._activation = act
        # model.int_blocks[0].layers_after_skip[0].dense_mlp[0]._activation = act
        # model.int_blocks[0].layers_after_skip[0].dense_mlp[1]._activation = act
        # model.int_blocks[0].layers_after_skip[1].dense_mlp[0]._activation = act
        # model.int_blocks[0].layers_after_skip[1].dense_mlp[1]._activation = act
        # model.int_blocks[0].atom_update.layers[0]._activation = act
        # model.int_blocks[0].atom_update.layers[1].dense_mlp[0]._activation = act
        # model.int_blocks[0].atom_update.layers[1].dense_mlp[1]._activation = act
        # model.int_blocks[0].atom_update.layers[2].dense_mlp[0]._activation = act
        # model.int_blocks[0].atom_update.layers[2].dense_mlp[1]._activation = act
        # model.int_blocks[0].atom_update.layers[3].dense_mlp[0]._activation = act
        # model.int_blocks[0].atom_update.layers[3].dense_mlp[1]._activation = act
        # model.int_blocks[0].concat_layer.dense._activation = act
        # model.int_blocks[0].residual_m[0].dense_mlp[0]._activation = act
        # model.int_blocks[0].residual_m[0].dense_mlp[1]._activation = act
        # model.int_blocks[1].dense_ca._activation = act
        # model.int_blocks[1].trip_interaction.dense_ba._activation = act
        # model.int_blocks[1].trip_interaction.down_projection._activation = act
        # model.int_blocks[1].trip_interaction.up_projection_ca._activation = act
        # model.int_blocks[1].trip_interaction.up_projection_ac._activation = act
        # model.int_blocks[1].layers_before_skip[0].dense_mlp[0]._activation = act
        # model.int_blocks[1].layers_before_skip[0].dense_mlp[1]._activation = act
        # model.int_blocks[1].layers_after_skip[0].dense_mlp[0]._activation = act
        # model.int_blocks[1].layers_after_skip[0].dense_mlp[1]._activation = act
        # model.int_blocks[1].layers_after_skip[1].dense_mlp[0]._activation = act
        # model.int_blocks[1].layers_after_skip[1].dense_mlp[1]._activation = act
        # model.int_blocks[1].atom_update.layers[0]._activation = act
        # model.int_blocks[1].atom_update.layers[1].dense_mlp[0]._activation = act
        # model.int_blocks[1].atom_update.layers[1].dense_mlp[1]._activation = act
        # model.int_blocks[1].atom_update.layers[2].dense_mlp[0]._activation = act
        # model.int_blocks[1].atom_update.layers[2].dense_mlp[1]._activation = act
        # model.int_blocks[1].atom_update.layers[3].dense_mlp[0]._activation = act
        # model.int_blocks[1].atom_update.layers[3].dense_mlp[1]._activation = act
        # model.int_blocks[1].concat_layer.dense._activation = act
        # model.int_blocks[1].residual_m[0].dense_mlp[0]._activation = act
        # model.int_blocks[1].residual_m[0].dense_mlp[1]._activation = act
        # model.int_blocks[2].dense_ca._activation = act
        # model.int_blocks[2].trip_interaction.dense_ba._activation = act
        # model.int_blocks[2].trip_interaction.down_projection._activation = act
        # model.int_blocks[2].trip_interaction.up_projection_ca._activation = act
        # model.int_blocks[2].trip_interaction.up_projection_ac._activation = act
        # model.int_blocks[2].layers_before_skip[0].dense_mlp[0]._activation = act
        # model.int_blocks[2].layers_before_skip[0].dense_mlp[1]._activation = act
        # model.int_blocks[2].layers_after_skip[0].dense_mlp[0]._activation = act
        # model.int_blocks[2].layers_after_skip[0].dense_mlp[1]._activation = act
        # model.int_blocks[2].layers_after_skip[1].dense_mlp[0]._activation = act
        # model.int_blocks[2].layers_after_skip[1].dense_mlp[1]._activation = act
        # model.int_blocks[2].atom_update.layers[0]._activation = act
        # model.int_blocks[2].atom_update.layers[1].dense_mlp[0]._activation = act
        # model.int_blocks[2].atom_update.layers[1].dense_mlp[1]._activation = act
        # model.int_blocks[2].atom_update.layers[2].dense_mlp[0]._activation = act
        # model.int_blocks[2].atom_update.layers[2].dense_mlp[1]._activation = act
        # model.int_blocks[2].atom_update.layers[3].dense_mlp[0]._activation = act
        # model.int_blocks[2].atom_update.layers[3].dense_mlp[1]._activation = act
        # model.int_blocks[2].concat_layer.dense._activation = act
        # model.int_blocks[2].residual_m[0].dense_mlp[0]._activation = act
        # model.int_blocks[2].residual_m[0].dense_mlp[1]._activation = act

    else:
        print("model activaiton not specified")
        assert False

    return