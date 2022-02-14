# AML Command Transfer (ACT)

ACT is a lightweight tool to transfer any command from the local machine to AML or
ITP, both of which are Azure Machine Learning services.

## Installation
1. install
   ```bash
   pip install --upgrade azureml-sdk
   pip install --upgrade --disable-pip-version-check --extra-index-url https://azuremlsdktestpypi.azureedge.net/K8s-Compute/D58E86006C65 azureml_contrib_k8s
   git clone https://github.com/microsoft/act.git
   cd act
   pip install -r requirements.txt
   python setup.py build develop
   ```

2. Setup azcopy

   Follow [this link](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)
   to download the azcopy and make sure that azcopy is downloaded to
   ~/code/azcopy/azcopy. That is, you can run the following to check if it is
   good.
   ```shell
   ~/code/azcopy/azcopy --version
   ```
   Make sure it is NOT version 8 or older.


3. Create the config file of `aux_data/configs/vigblob_account.yaml` for azure storage.
   The file format is
   ```yaml
   account_name: xxxx
   account_key: xxxx
   sas_token: ?xxxx
   container_name: xxxx
   ```
   The SAS token should start with the question mark.

4. Create the config file of `aux_data/aml/config.json` to specify the
   AML cluster information.
   ```json
   {
       "subscription_id": "xxxx",
       "resource_group": "xxxxx",
       "workspace_name": "xxxxx"
   }
   ```
   Make sure to have the double quotes to make it a valid json file.

5. Create the config file of `aux_data/aml/aml.yaml` to specify the submission
   related parameters. Here is one example.
   ```yaml
   azure_blob_config_file: null # no need to specify, legacy option
   datastore_name: null # no need to specify. legacy option
   # used to initialize the workspace
   aml_config: aux_data/aml/config.json 

   # the following is related with the job submission. If you don't use the
   # submission utility here, you can set any value

   config_param: 
      code_path:
          azure_blob_config_file: ./aux_data/configs/vigeastblob_account.yaml # the blob account information
          path: path/to/code.zip # where the zipped source code is
      # you can add multiple key-value pairs to configure the folder mapping.
      # Locally, if the folder name is A, and you want A to be a blobfuse
      # folder in the AML side, you need to set the key as A_folder. For
      # example, if the local folder is datasets, and you want datasets to be a
      # blobfuse folder in AML running, then add a pair with the key being
      # datasets_folder.
      data_folder:
          azure_blob_config_file: ./aux_data/configs/vigeastblob_account.yaml # the blob account information
          # after the source code is unzipped, this folder will be as $ROOT/data
          path: path/to/data
      output_folder:
          azure_blob_config_file: ./aux_data/configs/vigeastblob_account.yaml # the blob account information
          path: path/to/output # this folder will be as $ROOT/output
   # if False, it will use AML's PyTorch estimator, which is not heavily tested here
   use_custom_docker: true
   compute_target: NC24RSV3 
   # if it is the ITP cluster, please set it as true
   aks_compute: false
   docker:
       # the custom docker. If use_custom_docker is False, this will be ignored
       image: amsword/setup:py36pt16
   # any name to specify the experiment name.
   # better to have alias name as part of the experiment name since experiment
   # cannot be deleted and it is better to use fewer experiments
   experiment_name: experiment_name
   # if it is true, you need to run az login --use-device to authorize
   # before job submission. If you don't set it (default), it will prompt website to ask
   # you to do the authentication. It is recommmended to set it as True
   use_cli_auth: True
   # if it is true, it will spawn n processes on each node. n equals #gpu on
   # the node. otherwise, there will be only 1 process on each node. In
   # distributed training, if it is false, you might need to spawn n extra
   # processes by yourself. It is recommended to set it as true (default)
   multi_process: True
   gpu_per_node: 4
   env:
      # the dictionary of env will be as extra environment variables for the
      # job running. you can add multiple env here. Sometimes, the default
      # of NCCL_IB_DISABLE is '1', which will disable IB. Highly recommneded to
      # alwasy set it as '0', even when IB is not available.
      NCCL_IB_DISABLE: '0'
   # optionally, you can specify the option for zip command, which is used by
   # a init to compress the source folder and to upload it.
   zip_options:
       - '-x'
       - '\*src/py-faster-rcnn/\*'
       - '-x'
       - '\*src/CMC/\*'
   ```

6. Set an alias
   ```bash
   alis a='python -m act.aml_client '
   ```

## Job/Data Management
1. How to query the job status
   ```bash
   # the last parameter is the run id
   a query jianfw_1563257309_60ce2fc7
   a q jianfw_1563257309_60ce2fc7
   ```
   What it does
   1. Download the logs to the folder of `./assets/{RunID}`
   2. Print the last 100 lines of the log for ranker 0 if there is.
   3. Print the log paths so that you can copy/paste to open the log
   4. Print the meta data about the job, including status.
   One example of the output is

   ```bash
   0.2594)  loss_objectness: 0.0500 (0.0625)  loss_rpn_box_reg: 0.0438 (0.0539)  time: 0.9798 (0.9946)  data: 0.0058 (0.0134)  lr: 0.020000  max mem: 3831
   2019-07-16 20:41:29,098.098 trainer.py:138   do_train(): eta: 13:02:24  iter: 42800  speed: 16.1 images/sec  loss: 0.4821 (0.4971)  loss_box_reg: 0.1157 (0.1214)  loss_classifier: 0.2480 (0.2593)  loss_objectness: 0.0545 (0.0625)  loss_rpn_box_reg: 0.0383 (0.0539)  time: 0.9876 (0.9946)  data: 0.0056 (0.0133)  lr: 0.020000  max mem: 3831
   2019-07-16 20:43:07,526.526 trainer.py:138   do_train(): eta: 13:00:43  iter: 42900  speed: 16.3 images/sec  loss: 0.4585 (0.4971)  loss_box_reg: 0.1045 (0.1214)  loss_classifier: 0.2289 (0.2593)  loss_objectness: 0.0551 (0.0625)  loss_rpn_box_reg: 0.0506 (0.0539)  time: 0.9807 (0.9946)  data: 0.0058 (0.0133)  lr: 0.020000  max mem: 3831
   2019-07-16 20:44:46,805.805 trainer.py:138   do_train(): eta: 12:59:03  iter: 43000  speed: 16.1 images/sec  loss: 0.4569 (0.4970)  loss_box_reg: 0.1180 (0.1214)  loss_classifier: 0.2291 (0.2592)  loss_objectness: 0.0479 (0.0625)  loss_rpn_box_reg: 0.0436 (0.0539)  time: 0.9802 (0.9946)  data: 0.0058 (0.0133)  lr: 0.020000  max mem: 3831
   2019-07-16 14:30:26,592.592 aml_client.py:147      query(): log files:
   ['ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/70_driver_log_rank_0.txt',
    'ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/70_driver_log_rank_2.txt',
    ...
    'ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/55_batchai_execution-tvmps_e967edcdb10dd5e65827d221af1f6b246bb7d854790e27d26a677f78efe897ae_d.txt',
    'ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/55_batchai_stdout-job_prep-tvmps_e967edcdb10dd5e65827d221af1f6b246bb7d854790e27d26a677f78efe897ae_d.txt',
    'ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/55_batchai_stdout-job_prep-tvmps_3bbfd76728dd63d173c5cb80221dc4b244254a0fd864c695c8e70bf9460ac7ae_d.txt']
   2019-07-16 14:30:27,096.096 aml_client.py:38 print_run_info(): {'appID': 'jianfw_1563257309_60ce2fc7',
    'appID-s': 'e2fc7',
    'cluster': 'aml',
    'cmd': 'python src/qd/pipeline.py -bp '
           'YWxsX3Rlc3RfZGF0YToKLSB0ZXN0X2RhdGE6IGNvY28yMDE3RnVsbAogIHRlc3Rfc3BsaXQ6IHRlc3QKcGFyYW06CiAgSU5QVVQ6CiAgICBGSVhFRF9TSVpFX0FVRzoKICAgICAgUkFORE9NX1NDQUxFX01BWDogMS41CiAgICAgIFJBTkRPTV9TQ0FMRV9NSU46IDEuMAogICAgVVNFX0ZJWEVEX1NJWkVfQVVHTUVOVEFUSU9OOiB0cnVlCiAgTU9ERUw6CiAgICBGUE46CiAgICAgIFVTRV9HTjogdHJ1ZQogICAgUk9JX0JPWF9IRUFEOgogICAgICBVU0VfR046IHRydWUKICAgIFJQTjoKICAgICAgVVNFX0JOOiB0cnVlCiAgYmFzZV9scjogMC4wMgogIGRhdGE6IGNvY28yMDE3RnVsbAogIGRpc3RfdXJsX3RjcF9wb3J0OiAyMjkyMQogIGVmZmVjdGl2ZV9iYXRjaF9zaXplOiAxNgogIGV2YWx1YXRlX21ldGhvZDogY29jb19ib3gKICBleHBpZDogTV9CUzE2X01heEl0ZXI5MDAwMF9MUjAuMDJfU2NhbGVNYXgxLjVfRnBuR05fRlNpemVfUnBuQk5fSGVhZEdOX1N5bmNCTgogIGV4cGlkX3ByZWZpeDogTQogIGxvZ19zdGVwOiAxMDAKICBtYXhfaXRlcjogOTAwMDAKICBuZXQ6IGUyZV9mYXN0ZXJfcmNubl9SXzUwX0ZQTl8xeF90YmFzZQogIHBpcGVsaW5lX3R5cGU6IE1hc2tSQ05OUGlwZWxpbmUKICBzeW5jX2JuOiB0cnVlCiAgdGVzdF9kYXRhOiBjb2NvMjAxN0Z1bGwKICB0ZXN0X3NwbGl0OiB0ZXN0CiAgdGVzdF92ZXJzaW9uOiAwCnR5cGU6IHBpcGVsaW5lX3RyYWluX2V2YWxfbXVsdGkK',
    'elapsedTime': 15.27,
    'num_gpu': 8,
    'start_time': '2019-07-16T06:14:10.688519Z',
    'status': 'Canceled'}
   ```

2. How to abort/cancel a submitted job
   ```bash
   a abort jianfw_1563257309_60ce2fc7
   ```

3. How to resubmit a job
   ```bash
   a resubmit jianfw_1563257309_60ce2fc7
   a resubmit 60ce2fc7
   ```
   The resubmit here will first abort the existing job and then submit it.

4. How to submit the job

   The first step is to upload the code to azure blob by running the following
   command
   ```bash
   a init
   ```
   Whenever you want your new code change to take effect, you should run the above
   command. Otherwise, the job will use the previously uploaded code.
   To execute a command in AML, run the following:
   ```bash
   a submit cmd
   ```
   - if you want to run `nvidia-smi` in AML. The command is
   ```bash
   a submit nvidia-smi
   ```
   - If you want to run `python train.py --data voc20` in AML, the command
   will be
   ```bash
   a submit python train.py --data voc20
   ```
   - If you want to use 8 GPU, run the command like
   ```bash
   a -n 8 submit python train.py --data voc20
   ```
   `-n 8` should be placed before submit. Otherwise, it will think `-n 8` as
   part of the cmd
   - If `multi_process=true`, effectively it runs `mpirun --hostfile hostfile_contain_N_node_ips --npernode gpu_per_node cmd`
       - the number of nodes x gpu_per_node == the number of gpu requested
       - highly recommended for distributed training/inference
   - If `multi_process=false`, effectively it runs `mpirun --hostfile hostfile_contain_N_node_ips --npernode 1 cmd`
       - still, the number of nodes x gpu_per_node == the number of gpu requested
   - The rank needs to be figured out in the code generally. Internally, the
     service leverages the mpirun to launch the code. The rank or local rank
     can be figured out through mpirun-specific environment parameters.
     Sometimes, we also need to know the master node's IP, which can be figured
     out through
     ```python
     if 'AZ_BATCH_HOST_LIST' in os.environ:
         return get_aml_mpi_host_names()[0]
     elif 'AZ_BATCHAI_JOB_MASTER_NODE_IP' in os.environ:
         return os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
     ```
     There might be other variables as well to find the IP, but we will not
     list all of them here.

5. How to switch among multiple clusters
  For each cluster, it is recommended to have different configuration file. For
  example, we have two clusters: c1 and c2. Then, the two configuration files
  should be aux_data/aml/c1.yaml and aux_data/aml/c2.yaml. In this case, we can
  switch different clusters by the option of -c, e.g.
  ```bash
  a -c c1 submit ls
  a -c c2 submit nvidia-smi
  ```

6. Data management (optional)

   In the config file, we have a mapping of the local folder and the folder in
   the azure blob. Thus, we can upload and download the data based on this
   mapping. If the local folder is also a blobfuse folder, then there is no need
   to upload/download. Here, we mainly focus on the scenario where the local
   folder is not a blob fuse folder. Let's say the local folder name is `data`
   and we have an entry of `data_folder` in the config, which tells the data
   folder will be a blobfuse folder in AML env.
   - list the files starting with some prefix
     ```
     a ls data/voc20
     ```
     Note, the prefix here is `data/voc20`, which means we should have a
     definition of `data_folder` in the configuration
   - upload local file/folder of `data/voc20` to azure blob
     ```
     a u data/voc20
     ```
   - download the file/folder of `data/coco` from blob to local folder
     ```
     a d data/coco
     ```
     Note
     -  `u` means upload; `d` means download
     - it will automatically identify if it is a file or folder. Thus, there is no
       need to specify special parameters here.
   - delete a file or folder in the blob defined by the clsuter config
     ```
     a rm data/coco
     ```
     Be careful as you can not revert
     this operation or cannot recover the data if the deletion is a mistake.
   - transfer the file or folder between two blobs
     ```
     a -c eu -f we3v32 u data/voc20
     ```
     Here, `-c` means current cluster name. In this case, it will by default
     find the config through `aux_data/aml/eu.yaml`. `-f` means `from cluster`,
     which means the data source. Each cluster has a definition of the blob
     information. Thus, this tool can figure out all details to transfer the
     data from another cluster's setting to this cluster's blob setting. It
     will also automatically detect whether to take it like a folder or a file.
   
   

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
