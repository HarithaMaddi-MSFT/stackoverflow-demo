def register_data():
    import azureml.core
    from azureml.core import Workspace, Datastore, Dataset
    
    workspace = Workspace.from_config()
    ds = workspace.get_default_datastore()

    try:
        stackoverflow_dataset = workspace.datasets['stackoverflow']
        raw_dataset = workspace.datasets['stackoverflow']
    except:
        ds.upload_files(files = ['data/train.csv', 'data/test.csv','data/valid.csv','data/classes.txt'], target_path = 'data', show_progress = True)
        dataset = Dataset.File.from_files(path = [(ds, 'data')])
        dataset = dataset.register(workspace = workspace, name = 'stackoverflow', description='training, test and validation files', create_new_version=True)
        stackoverflow_dataset = workspace.datasets['stackoverflow']
        raw_dataset = workspace.datasets['stackoverflow']

    try:
        azure_support_volume_timeseries_train = workspace.datasets['azure_support_volume_timeseries_train']
        azure_support_volume_timeseries_test = workspace.datasets['azure_support_volume_timeseries_test']
    except:
        time_column_name = 'date'
        ds.upload_files(files = ['data/azure_support_volume_timeseries_train.csv', 'data/azure_support_volume_timeseries_test.csv'], target_path = 'timeseries_data', show_progress = True)
        
        train_dataset = Dataset.Tabular.from_delimited_files(path=[(ds, 'timeseries_data/azure_support_volume_timeseries_train.csv')]).with_timestamp_columns(fine_grain_timestamp=time_column_name)
        azure_support_volume_timeseries_train = train_dataset.register(workspace = workspace, name = 'azure_support_volume_timeseries_train', description='azure support volume timeseries training data', create_new_version=True)

        test_dataset = Dataset.Tabular.from_delimited_files(path=[(ds, 'timeseries_data/azure_support_volume_timeseries_test.csv')]).with_timestamp_columns(fine_grain_timestamp=time_column_name)      
        azure_support_volume_timeseries_test = train_dataset.register(workspace = workspace, name = 'azure_support_volume_timeseries_test', description='azure support volume timeseries testing data', create_new_version=True)

    return stackoverflow_dataset, raw_dataset, azure_support_volume_timeseries_train, azure_support_volume_timeseries_test

def startup():
    import azureml.core
    from azureml.core import Run, Workspace, Experiment
    from azureml.pipeline.core import PublishedPipeline
    from azureml.core import Datastore, Dataset
    import pandas as pd
    print("SDK version:", azureml.core.VERSION)
    pd.set_option('display.max_colwidth', 120)
    from azureml.core import Datastore, Dataset

    workspace = Workspace.from_config()
    
    ds = workspace.get_default_datastore()
    
    #target_column_name = 'volume'
    #time_column_name = 'date'
    #time_series_id_column_names = 'team_tag'

    experiment_name = 'azure-stackoverflow-classifier' 
    experiment = Experiment(workspace, name=experiment_name)
    train = pd.read_csv('./data/train.csv', names=['ID', 'IssueTitle', 'Label'])

    try:
    
        run = Run(experiment, 'azure-stackoverflow-classifier_1592684426_3767f390')
        hd_run = Run(experiment, 'HD_ddfd3027-4b17-4afd-a42f-cec512ec544b')
        aks_service = workspace.webservices['stackoverflow-classifier']
    
        pipelines = PublishedPipeline.list(workspace)
        published_pipeline = pipelines[0]
    
    except:
        print("demo not initialized ... to speed up demo, after you have run through demo script all the way, set the values for the Run, HD_Run and AKS Service to fetch from existing entities instead of running realtime")
        run = ""
        hd_run = ""
        aks_service = ""    
        published_pipeline = ""

    stackoverflow_dataset, raw_dataset, azure_support_volume_timeseries_train, azure_support_volume_timeseries_test = register_data()

    return ds, run, hd_run, aks_service, published_pipeline, stackoverflow_dataset, raw_dataset, train, azure_support_volume_timeseries_train, azure_support_volume_timeseries_test

