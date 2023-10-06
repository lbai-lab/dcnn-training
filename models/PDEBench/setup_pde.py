"""
Only minor changes to enable activation replacement and batch normalization
This source code is adapted from PDEBench:
https://github.com/pdebench/PDEBench/tree/main
under the MIT license found in:
https://github.com/pdebench/PDEBench/blob/main/LICENSE.txt
"""

import deepxde as dde
import numpy as np
import torch

from models.PDEBench.pde_definitions import *
from models.PDEBench.backboneNN import backboneNN
from models.PDEBench.mlp import DeepXDE_FNN
from datasets.PDEBench.PDEBenchDataset import *
from typing import Tuple

def setup_diffusion_sorption(filename, seed, root_path, act, backbone, batchnorm):
    # TODO: read from dataset config file
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 500.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    D = 5e-4

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    bc_d = dde.icbc.DirichletBC(
        geomtime,
        lambda x: 1.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
    )

    def operator_bc(inputs, outputs, X):
        # compute u_t
        du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return outputs - D * du_x

    bc_d2 = dde.icbc.OperatorBC(
        geomtime,
        operator_bc,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0),
    )

    dataset = PINNDatasetDiffSorption(filename, seed, root_path=root_path)

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1])

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_sorption,
        [ic, bc_d, bc_d2, bc_data],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )

    dim_hidden, depth = backboneNN[backbone]
    if act:
        print("replace activations")
        activation = [act for _ in range(depth)] + [None]
    else:
        activation = "tanh"

    net = DeepXDE_FNN([2] + [dim_hidden] * depth + [1], activation, "Glorot normal", batchnorm=batchnorm)

    def transform_output(x, y):
        return torch.relu(y)

    net.apply_output_transform(transform_output)

    model = dde.Model(data, net)

    return model, dataset

def setup_diffusion_reaction(filename, seed, root_path, act, backbone, batchnorm):
    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-1, -1), (1, 1))
    timedomain = dde.geometry.TimeDomain(0, 5.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

    dataset = PINNDatasetDiffReact(filename, seed, root_path=root_path)
    initial_input, initial_u, initial_v = dataset.get_initial_condition()

    ic_data_u = dde.icbc.PointSetBC(initial_input, initial_u, component=0)
    ic_data_v = dde.icbc.PointSetBC(initial_input, initial_v, component=1)

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]
    bc_data_u = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)
    bc_data_v = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[2], component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_reaction,
        [bc, ic_data_u, ic_data_v, bc_data_u, bc_data_v],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )

    dim_hidden, depth = backboneNN[backbone]
    if act:
        print("replace activations")
        activation = [act for _ in range(depth)] + [None]
    else:
        activation = "tanh"

    net = DeepXDE_FNN([3] + [dim_hidden] * depth + [2], activation, "Glorot normal", batchnorm=batchnorm)
    model = dde.Model(data, net)

    return model, dataset

def setup_swe_2d(filename, seed, root_path, act, backbone, batchnorm) -> Tuple[dde.Model, PINNDataset2D]:

    dataset = PINNDatasetRadialDambreak(filename, seed, root_path=root_path)

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-2.5, -2.5), (2.5, 2.5))
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic_h = dde.icbc.IC(
        geomtime,
        dataset.get_initial_condition_func(),
        lambda _, on_initial: on_initial,
        component=0,
    )
    ic_u = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=1
    )
    ic_v = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=2
    )

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)

    data = dde.data.TimePDE(
        geomtime,
        pde_swe2d,
        [bc, ic_h, ic_u, ic_v, bc_data],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )

    dim_hidden, depth = backboneNN[backbone]
    if act:
        print("replace activations")
        activation = [act for _ in range(depth)] + [None]
    else:
        activation = "tanh"

    net = DeepXDE_FNN([3] + [dim_hidden] * depth + [3], activation, "Glorot normal", batchnorm=batchnorm)
    model = dde.Model(data, net)

    return model, dataset

def _boundary_r(x, on_boundary, xL, xR):
    return (on_boundary and np.isclose(x[0], xL)) or (on_boundary and np.isclose(x[0], xR))

def setup_pde1D(filename="1D_Advection_Sols_beta0.1.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=1,
                hidden_ch=40,
                xL=0.,
                xR=1.,
                if_periodic_bc=True,
                aux_params=[0.1],
                act = None, 
                backbone ="40-6",
                batchnorm = False):

    # TODO: read from dataset config file
    geom = dde.geometry.Interval(xL, xR)
    boundary_r = lambda x, on_boundary: _boundary_r(x, on_boundary, xL, xR)
    if filename[0] == 'R':
        timedomain = dde.geometry.TimeDomain(0, 1.0)
        pde = lambda x, y : pde_diffusion_reaction_1d(x, y, aux_params[0], aux_params[1])
    else:
        if filename.split('_')[1][0]=='A':
            timedomain = dde.geometry.TimeDomain(0, 2.0)
            pde = lambda x, y: pde_adv1d(x, y, aux_params[0])
        elif filename.split('_')[1][0] == 'B':
            timedomain = dde.geometry.TimeDomain(0, 2.0)
            pde = lambda x, y: pde_burgers1D(x, y, aux_params[0])
        elif filename.split('_')[1][0]=='C':
            timedomain = dde.geometry.TimeDomain(0, 1.0)
            pde = lambda x, y: pde_CFD1d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset1Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    if filename.split('_')[1][0] == 'C':
        ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,0].unsqueeze(1), component=0)
        ic_data_v = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,1].unsqueeze(1), component=1)
        ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,2].unsqueeze(1), component=2)
    else:
        ic_data_u = dde.icbc.PointSetBC(initial_input.cpu(), initial_u, component=0)
    
    # prepare boundary condition
    if if_periodic_bc:
        if filename.split('_')[1][0] == 'C':
            bc_D = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            bc_V = dde.icbc.PeriodicBC(geomtime, 1, boundary_r)
            bc_P = dde.icbc.PeriodicBC(geomtime, 2, boundary_r)

            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_d, ic_data_v, ic_data_p, bc_D, bc_V, bc_P],
                num_domain=1000,
                num_boundary=1000,
                num_initial=5000,
            )
        else:
            bc = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_u, bc],
                num_domain=1000,
                num_boundary=1000,
                num_initial=5000,
            )
    else:
        ic = dde.icbc.IC(
            geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
        )
        bd_input, bd_uL, bd_uR = dataset.get_boundary_condition()
        bc_data_uL = dde.icbc.PointSetBC(bd_input.cpu(), bd_uL, component=0)
        bc_data_uR = dde.icbc.PointSetBC(bd_input.cpu(), bd_uR, component=0)

        data = dde.data.TimePDE(
            geomtime,
            pde,
            [ic, bc_data_uL, bc_data_uR],
            num_domain=1000,
            num_boundary=1000,
            num_initial=5000,
        )
    dim_hidden, depth = backboneNN[backbone]
    
    if act:
        activation = [act for _ in range(depth)] + [None]
    else:
        activation = "tanh"

    net = DeepXDE_FNN([input_ch] + [dim_hidden] * depth + [output_ch], activation, "Glorot normal", batchnorm=batchnorm)
    model = dde.Model(data, net)

    return model, dataset

def setup_CFD2D(filename="2D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=4,
                hidden_ch=40,
                xL=0.,
                xR=1.,
                yL=0.,
                yR=1.,
                if_periodic_bc=True,
                aux_params=[1.6667], 
                backbone="40-6"):

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-1, -1), (1, 1))
    timedomain = dde.geometry.TimeDomain(0., 1.0)
    pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset2Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
    ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
    ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
    ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
    # prepare boundary condition
    bc = dde.icbc.PeriodicBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_d, ic_data_vx, ic_data_vy, ic_data_p],#, bc],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def setup_CFD3D(filename="3D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=4,
                hidden_ch=40,
                aux_params=[1.6667],
                backbone="40-6"):

    # TODO: read from dataset config file
    geom = dde.geometry.Cuboid((0., 0., 0.), (1., 1., 1.))
    timedomain = dde.geometry.TimeDomain(0., 1.0)
    pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset3Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
    ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
    ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
    ic_data_vz = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
    ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,4].unsqueeze(1), component=4)
    # prepare boundary condition
    bc = dde.icbc.PeriodicBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_d, ic_data_vx, ic_data_vy, ic_data_vz, ic_data_p, bc],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset