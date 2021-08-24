import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--suffix',type=str,default='_filtered_we', help='data file suffix')
parser.add_argument('--suffix_train',type=str,default='_filtered_we', help='data file suffix')
parser.add_argument('--eRec',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,help='')
parser.add_argument('--plotheatmap',type=str,default='True',help='')
parser.add_argument('--save',type=str,default='results/model',help='save path')


args = parser.parse_args()


def plot_results(model_name, mae_test, detector=None):
    if detector:
        for t_idx in range(mae_test.shape[1]):
            # for d_idx in range(mae_test.shape[2]):
            d_idx = detector
            plt.title(f'{model_name}: MAE Detector {d_idx}')
            plt.xlabel('time-steps')
            plt.ylabel('mae (ml/h)')
            plt.plot(mae_test[:, t_idx, d_idx].detach().numpy())
            plt.savefig(f'results/{model_name}_mae_t{t_idx}.jpg')
            plt.show()
    else:
        for t_idx in range(mae_test.shape[1]):
            for d_idx in range(mae_test.shape[2]):
                plt.title(f'MAE Detector {d_idx}')
                plt.xlabel('time-steps')
                plt.ylabel('mae (ml/h)')
                plt.plot(mae_test[:, t_idx, d_idx].detach().numpy())
                plt.savefig(f'results/{model_name}_mae_t{t_idx}.jpg')
                plt.show()
    return


def print_results(model_name, mse_test, mae_test):
    loss_type = 'MAE'  # MAE
    # Final Results
    result_folder = 'results/'
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(result_folder + f'final_results_{model_name}.txt', 'w') as filehandle:
        sys.stdout = filehandle  # Change the standard output to the file we created.
        print('1. General mean')
        print(f'Final MSE = {torch.mean(mse_test)}')
        print(f'Final MAE = {torch.mean(mae_test)}')

        print('2. By detector')
        print(f'Final MSE = {torch.mean(mse_test, (0, 1), keepdim=True)}')
        print(f'Final MAE = {torch.mean(mae_test, (0, 1), keepdim=True)}')

        print('3. By time')
        print(f'Final MSE = {torch.mean(mse_test, (0, 2), keepdim=True)}')
        print(f'Final MAE = {torch.mean(mae_test, (0, 2), keepdim=True)}')
        tab = PrettyTable()
        tab.field_names = [f"time_{i + 1}" for i in range(mae_test.shape[1])]
        out = torch.mean(mae_test, (0, 2))
        tab.add_row(out.detach().numpy())
        print(f'{loss_type} per sequence time-step:')
        print(tab, '\n')

        print('4. By detector & time')
        print(f'Final MSE = {torch.mean(mse_test, (0), keepdim=True)}')
        print(f'Final MAE = {torch.mean(mae_test, (0), keepdim=True)}')
        tab = PrettyTable()
        tab.field_names = [f"time_{i + 1}" for i in range(mae_test.shape[1])]
        out = torch.mean(mae_test, 0)
        tab.add_rows(out.transpose(0, 1).detach().numpy())
        print(f'{loss_type} per sensor per sequence time-step')
        print(tab, '\n')

        sys.stdout = original_stdout  # Reset the standard output to its original value

    with open(result_folder + f'final_results_{model_name}.txt', 'r') as filehandle:
        print(filehandle.read())

    return


def main():
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    eR_seq_size = 24  # 24
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size,
                                   eRec=args.eRec, eR_seq_size=eR_seq_size, suffix=args.suffix_train)
    scaler = dataloader['scaler']

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size,
                                   eRec=args.eRec, eR_seq_size=eR_seq_size, suffix=args.suffix, scaler=scaler)

    blocks = int(dataloader[f'x_test{args.suffix}'].shape[1] / 3)  # Every block reduce the input sequence size by 3.
    print(f'blocks = {blocks}')

    if args.eRec:
        error_size = 6
        model = eRGwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj,
                        adjinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid,
                        dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16,
                        blocks=blocks, error_size=error_size)
    else:
        model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj,
                        adjinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid,
                        dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16,
                        blocks=blocks)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(device)))
    model.eval()

    print('model load successfully')
    outputs = []
    realy = torch.Tensor(dataloader[f'y_test{args.suffix}']).to(device)
    if args.eRec:
        realy = realy.transpose(0, 1)[-1, :, :, :, :]
    realy = realy.transpose(1,3)[:,0,:,:]
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        if args.eRec:
            testx = testx.transpose(0, 1)
            testy = testy.transpose(0, 1)
        testx = testx.transpose(-3, -1)
        testy = testy.transpose(-3, -1)
        with torch.no_grad():
            if args.eRec:
                preds = model(testx, testy[:, :, 0:1, :, :], scaler).transpose(1,3)
            else:
                preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    print(f'yhat before shape = {yhat.shape}')
    yhat = yhat[:realy.size(0),...]
    print(f'yhat shape = {yhat.shape}')


    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))

    mse = nn.MSELoss(reduction='none')
    mae = nn.L1Loss(reduction='none')
    pred = scaler.inverse_transform(yhat)
    loss_mse = mse(pred, realy).transpose(1,2)
    loss_mae = mae(pred, realy).transpose(1,2)
    print(f'loss_mae shape = {loss_mae.shape}')
    if args.eRec:
        model_name = f'gwnet{args.suffix}'
    else:
        model_name = f'eRgwnet{args.suffix}'
    print_results(model_name, loss_mse, loss_mae)
    plot_results(model_name, loss_mae, detector=1)

    if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./emb"+ '.pdf')

    y12 = realy[:,-1,-1].cpu().detach().numpy()
    yhat12 = scaler.inverse_transform(yhat[:,-1,-1]).cpu().detach().numpy()

    y3 = realy[:,-1,2].cpu().detach().numpy()
    yhat3 = scaler.inverse_transform(yhat[:,-1,2]).cpu().detach().numpy()

    df2 = pd.DataFrame({'real12':y12,'pred12':yhat12, 'real3': y3, 'pred3':yhat3})
    df2.to_csv('./wave.csv',index=False)


if __name__ == "__main__":
    main()
