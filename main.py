import argparse
import readfile
import hmm
import part4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--part',
    required=True,
    help='Possible parts: 1, 2, 3, 4')

    parser.add_argument('--datasets',
    required=True,
    help='Input datasets ES/RU/ES-test/RU-test. Separate by comma if there are multiple input datasets. Ensure that datasets are stored in \'Datasets/\'')

    parser.add_argument('--k_num',
    default = 5,
    help='Used when running part 3. Defines the k-th best sequence to obtain. Defaults to 5.')

    parser.add_argument('--epochs',
    default=10,
    help='Used when running part 4. Defaults to 10.')

    args = parser.parse_args()

    dataset_list = args.datasets.split(',')
    datasets = [f'Datasets/{dataset}' for dataset in dataset_list]

    for ds_path in datasets:
        if ds_path[-5:] == '-test':
            train_dataset = readfile.read_train_file(f'{ds_path[:-5]}/train')
            dev_dataset = readfile.read_dev_in_file(f'{ds_path}/test.in')
        else:
            train_dataset = readfile.read_train_file(f'{ds_path}/train')
            dev_dataset = readfile.read_dev_in_file(f'{ds_path}/dev.in')

        if args.part == '1':
            model = hmm.HMM()
            model.train(train_dataset)
            model.predict_part1(dev_dataset).write_preds(f'{ds_path}/dev.p{args.part}.out')
        
        elif args.part == '2':
            model = hmm.HMM()
            model.train(train_dataset)
            model.predict_part2(dev_dataset).write_preds(f'{ds_path}/dev.p{args.part}.out')

        elif args.part == '3':
            model = hmm.HMM()
            model.train(train_dataset)
            model.predict_part3(dev_dataset).write_preds(f'{ds_path}/dev.p{args.part}.out')

        elif args.part == '4':
            model = part4.StructuredPerceptron()
            model.train(train_dataset, args.epochs)
            model.predict(dev_dataset).write_preds(f'{ds_path}/dev.p{args.part}.out')

        else:
            raise argparse.ArgumentError(message='Input valid parameters for --part.')

    

