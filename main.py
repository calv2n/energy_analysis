import utils
import config

if __name__ == "__main__":
    for i, dataset_config in enumerate(config.DATASETS):
        dataset = dataset_config['name']
        target = dataset_config['target']
        parser = dataset_config['parser']
        hyperparameters = config.HYPERPARAMETERS[i]

        results_rolling = utils.find_best_model_rolling(
            dataset=dataset, 
            target=target, 
            p_candidates=config.P_CANDIDATES, 
            prop_test=config.TEST_PROP,
            parser=parser, 
            hyperparameters=hyperparameters
        )
        utils.plot_best_rolling(results_rolling, dataset_name=dataset)
        print()

        results_uptodate = utils.find_best_model_uptodate(
            dataset=dataset, 
            target=target, 
            p_candidates=config.P_CANDIDATES, 
            prop_test=config.TEST_PROP,
            parser=parser,
            hyperparameters=hyperparameters
        )
        utils.plot_best_uptodate(results_uptodate, dataset_name=dataset)
        print()
    