import os

class S3Sync:
    """  
    This class provides methods to upload a folder to an S3 bucket and download 
    a folder from an S3 bucket using the AWS CLI. 
    """
    def sync_folder_to_s3(self,folder,aws_bucket_url):
        """ 
        This method uploads the contents of a local folder to the specified 
        Amazon S3 bucket using the `aws s3 sync` command.

        Args:
            folder (str): The path to the local folder to be uploaded.
            aws_bucket_url (str): The S3 bucket URL where the folder will be uploaded. 
        """
        command=f"aws s3 sync {folder} {aws_bucket_url} "
        os.system(command)

    def sync_folder_from_s3(self,folder,aws_bucket_url):
        """ 
        This method downloads the contents of the specified Amazon S3 bucket 
        to a local folder using the `aws s3 sync` command.

        Args:
            folder (str): The path to the local folder where the contents will be downloaded.
            aws_bucket_url (str): The S3 bucket URL from where the folder will be downloaded. 
        """
        command=f"aws s3 sync {aws_bucket_url} {folder} "
        os.system(command)