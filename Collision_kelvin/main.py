import torch
import argparse
from time import time
from utils.solver import Solver
from utils.seed import fixSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        """ 运行参数 """
        parser.add_argument('--model', type=str, default='TRANSFORMER', help='backbone network',
                            choices=['LSTM', 'TRANSFORMER','CDEA', 'BAYES_LSTM'])
        parser.add_argument('--only_test', default=False, action='store_true', help='only test the model')

        """ 数据参数 """
        parser.add_argument('--data_path', type=str, default='./dataset/', help='path to dataset')
        parser.add_argument('--save_path', type=str, default='./results/checkpoint/', help='path to save model')
        parser.add_argument('--dataset_train', type=str, default='train_data.csv', help='dataset name')
        parser.add_argument('--dataset_test', type=str, default='test_data.csv', help='dataset name')
        parser.add_argument('--ratio_split', type=float, nargs='+', default=[0.7, 0.15, 0.15],
                            help='train,val,test data ratio')
        parser.add_argument('--n_latest_cdms', type=int, default=13,
                            help='number of latest CDMs to keep per event for sequence construction')

        """ LSTM模型超参 """
        parser.add_argument('--hidden_size', type=int, default=64, help='hidden units for LSTM backbone')
        parser.add_argument('--num_layers', type=int, default=3, help='LSTM layer count')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate for LSTM')

        """Transformer 超参"""
        parser.add_argument('--d_model', type=int, default=128, help='Transformer embedding dim')
        parser.add_argument('--nhead', type=int, default=4, help='Transformer number of attention heads')
        parser.add_argument('--num_layers_tf', type=int, default=3, help='Transformer encoder layer count')
        parser.add_argument('--dim_ff', type=int, default=256, help='Transformer feedforward dim')
        parser.add_argument('--tf_dropout', type=float, default=0.15, help='dropout rate for Transformer')
        
        """CDEA 超参"""
        parser.add_argument('--cdea_layers', type=int, default=10, help='number of CDEA blocks')
        parser.add_argument('--cdea_heads_row', type=int, default=4, help='row-wise attention heads')
        parser.add_argument('--cdea_heads_col', type=int, default=4, help='col-wise attention heads')
        parser.add_argument('--cdea_k_neighbors', type=int, default=8, help='KNN neighbors for physical graph')

        """Bayes_LSTM 超参"""
        parser.add_argument('--bayes_hidden_size', type=int, default=256, help='hidden units for Bayes LSTM')
        parser.add_argument('--bayes_num_layers', type=int, default=2, help='Bayes LSTM layer count')
        parser.add_argument('--bayes_dropout', type=float, default=0.2, help='dropout rate for Bayes LSTM')
        
        """ 新增损失函数参数 """
        parser.add_argument('--lambda_rank', type=float, default=0.1, 
                            help='weight for RankLoss (0.0 to disable, recommended: 0.1)')
        parser.add_argument('--rank_margin', type=float, default=0.0, 
                            help='margin for RankLoss (usually 0.0 is fine)')
        
        parser.add_argument('--use_supcr', action='store_true', default=True, 
                            help='whether to use SupConRegressionLoss (requires model to support return_feat)')
        parser.add_argument('--lambda_sup', type=float, default=0.1, 
                            help='weight for SupConRegressionLoss (0.0 to disable, recommended: 0.5)')
        parser.add_argument('--sup_temp', type=float, default=0.1, 
                            help='temperature for SupConRegressionLoss (lower = harder contrast)')
        parser.add_argument('--sup_sigma', type=float, default=2.0, 
                            help='sigma for SupConRegressionLoss label similarity (adjust based on label range)')
        
        """ 优化参数 """
        parser.add_argument('--itr', type=int, default=1, help='number of iterations')  ########## 运行次数
        parser.add_argument('--epoch', type=int, default=150, help='number of epochs')
        parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

        """ GPU 参数 """
        parser.add_argument('--seed', type=int, default=43, help='random seed')  # 42
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
        parser.add_argument('--device', type=int, default=0, help='device id')

        args = parser.parse_args()
        if len(args.ratio_split) != 3:
            raise ValueError('ratio_split 需要包含 [train, val, test] 三个比例')
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        print('\n=====================Args========================')
        print(args)
        print('=================================================\n')
        fixSeed(args.seed)

        for ii in range(args.itr):

            setting = f"{args.model}_{args.seed}_{ii}"

            print(f"\n>>>>>>>>  initing : {setting}  <<<<<<<<\n")
            solver = Solver(args, setting)

            if not args.only_test:
                print(f"\n>>>>>>>>  training : {setting}  <<<<<<<<\n")
                start = time()
                epoch, result_metrics_val = solver.train()
                train_time = (time() - start) / epoch
                print(f"Training Time: {train_time:.4f}s")
                # 保存metrics到csv
                solver._save_metrics_to_csv(result_metrics_val)

            print(f"\n>>>>>>>>  testing : {setting}  <<<<<<<<\n")
            start = time()
            res, result_metrics_test = solver.test()
            test_time = time() - start
            print(f"Testing Time: {test_time:.4f}s")

        torch.cuda.empty_cache()
        print('Done!')