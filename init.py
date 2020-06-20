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
    #datastore_name = 'tfworld'
    #container_name = 'azure-service-classifier'
    #account_name = 'johndatasets'
    #sas_token = '?sv=2019-02-02&ss=bfqt&srt=sco&sp=rl&se=2021-06-02T03:40:25Z&st=2020-03-09T19:40:25Z&spr=https&sig=bUwK7AJUj2c%2Fr90Qf8O1sojF0w6wRFgL2c9zMVCWNPA%3D'

    #datastore = Datastore.register_azure_blob_container(workspace=workspace, 
    #                                                    datastore_name=datastore_name, 
    #                                                    container_name=container_name,
    #                                                    account_name=account_name, 
    #                                                    sas_token=sas_token, overwrite=True)
    
    
    ds = workspace.get_default_datastore()
    #ds.upload_files(files = ['./data/train.csv', './data/test.csv','./data/valid.csv','./data/classes.txt'],
    #                   target_path = 'data',
    #                   show_progress = True)
    
    #dataset = Dataset.File.from_files(path = [(ds, 'data')])

    #dataset = dataset.register(workspace = workspace,
    #                       name = 'stackoverflow',
    #                       description='training, test and validation files',
    #                       create_new_version=True)
    
    experiment_name = 'azure-stackoverflow-classifier' 
    experiment = Experiment(workspace, name=experiment_name)
    train = pd.read_csv('./data/train.csv', names=['ID', 'IssueTitle', 'Label'])
    run = Run(experiment, 'azure-stackoverflow-classifier_1591901512_a874f89a')
    hd_run = Run(experiment, 'HD_8fd1053a-b59a-41c9-b863-a6cd42a4b5c6')
    aks_service = workspace.webservices['stackoverflow-classifier']
    
    pipelines = PublishedPipeline.list(workspace)
    published_pipeline = pipelines[0]
    
    stackoverflow_dataset = workspace.datasets['stackoverflow']
    raw_dataset = workspace.datasets['stackoverflow']
    
    ## For AutoML demo
    target_column_name = 'volume'
    time_column_name = 'date'
    time_series_id_column_names = 'team_tag'
    train_dataset = Dataset.Tabular.from_delimited_files(path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/ForecastingDemo/azure_support_volume_timeseries_train.csv").with_timestamp_columns(fine_grain_timestamp=time_column_name) 
    test_dataset = Dataset.Tabular.from_delimited_files(path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/ForecastingDemo/azure_support_volume_timeseries_test.csv").with_timestamp_columns(fine_grain_timestamp=time_column_name) 

    
    return ds, run, hd_run, aks_service, published_pipeline, stackoverflow_dataset, raw_dataset, train, train_dataset, test_dataset

startup()


