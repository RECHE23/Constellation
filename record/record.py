# -*- coding: utf-8 -*-

from poutyne.framework import Model
from . import *


def start_record(models, datasets, device, batch_size, representations, epochs,
                 metrics):
    for model_name, model in models.items():
        for ds in datasets:
            print(f'{model_name} with {ds.name} :\n')
            ds.to(device)
            splitter = DataSplit(ds, test_train_split=TEST_TRAIN_SPLIT,
                                 val_train_split=VAL_TRAIN_SPLIT, shuffle=True)
            loaders = splitter.get_split(batch_size=batch_size)
            train_loader, valid_loader, test_loader = loaders
            net = model['network'](dataset=ds)
            optimizer = model['optimizer'][0](net.parameters(),
                                              **model['optimizer'][1])
            base_callback = BaseCB(model_name=model_name,
                                   dataset_name=ds.name,
                                   records_path=RECORDS_PATH)
            callbacks = [
                base_callback,
                DatasetCB(base_callback=base_callback, dataset=ds, batch_size=batch_size),
                MetricsCB(base_callback=base_callback, batch_metrics=metrics),
                LearningRateCB(base_callback=base_callback),
                DecisionBoundariesCB(base_callback=base_callback, dataset=ds,
                                     representations=representations),
                WeightsBiasesCB(base_callback=base_callback)
            ]
            if model['scheduler'] is not None:
                scheduler = model['scheduler'][0](**model['scheduler'][1])
                callbacks.append(scheduler)
            batch_metrics = [metrics_list[metric] for metric in metrics]
            poutyne_model = Model(net, optimizer, model['loss function'],
                                  batch_metrics=batch_metrics)
            poutyne_model.to(device)
            poutyne_model.fit_generator(train_loader, valid_loader,
                                        epochs=epochs,
                                        callbacks=callbacks)
            test_loss, test_acc = poutyne_model.evaluate_generator(test_loader)
            print(f'Test:\n\tLoss: {test_loss}\n\tAccuracy: {test_acc}\n')
