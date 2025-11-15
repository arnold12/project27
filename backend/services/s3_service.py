"""
AWS S3 service for storing and retrieving documents.
"""
import os
import boto3
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO
from pathlib import Path

from backend.core.config import settings
from backend.core.utils import ensure_directory


class S3Service:
    """Service for interacting with AWS S3."""
    
    def __init__(self):
        """Initialize S3 client."""
        self.bucket_name = settings.aws_bucket
        self.region = settings.aws_region
        
        # Initialize S3 client
        s3_kwargs = {
            "region_name": self.region,
        }
        
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            s3_kwargs.update({
                "aws_access_key_id": settings.aws_access_key_id,
                "aws_secret_access_key": settings.aws_secret_access_key,
            })
        
        if settings.aws_endpoint_url:
            s3_kwargs["endpoint_url"] = settings.aws_endpoint_url
        
        self.s3_client = boto3.client("s3", **s3_kwargs)
        
        # Ensure bucket exists (create if it doesn't)
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Ensure the S3 bucket exists, create if it doesn't."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            # Bucket doesn't exist, try to create it
            try:
                if self.region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.region}
                    )
            except ClientError as e:
                # If creation fails, we'll handle it in upload operations
                pass
    
    def upload_file(
        self,
        file_path: str,
        s3_key: str,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Upload a file to S3.
        
        Args:
            file_path: Local file path
            s3_key: S3 object key
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = {
                    str(k): str(v) for k, v in metadata.items()
                }
            
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            return True
        except ClientError as e:
            print(f"Error uploading file to S3: {e}")
            return False
    
    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        s3_key: str,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Upload a file-like object to S3.
        
        Args:
            file_obj: File-like object
            s3_key: S3 object key
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = {
                    str(k): str(v) for k, v in metadata.items()
                }
            
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            return True
        except ClientError as e:
            print(f"Error uploading file to S3: {e}")
            return False
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            ensure_directory(os.path.dirname(local_path))
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_path
            )
            return True
        except ClientError as e:
            print(f"Error downloading file from S3: {e}")
            return False
    
    def get_file_content(self, s3_key: str) -> Optional[bytes]:
        """
        Get file content from S3 as bytes.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            File content as bytes, or None if error
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response["Body"].read()
        except ClientError as e:
            print(f"Error getting file from S3: {e}")
            return None
    
    def get_text_content(self, s3_key: str) -> Optional[str]:
        """
        Get text content from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Text content, or None if error
        """
        content = self.get_file_content(s3_key)
        if content:
            return content.decode("utf-8")
        return None
    
    def save_text_content(self, text: str, s3_key: str) -> bool:
        """
        Save text content to S3.
        
        Args:
            text: Text content to save
            s3_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=text.encode("utf-8"),
                ContentType="text/plain"
            )
            return True
        except ClientError as e:
            print(f"Error saving text to S3: {e}")
            return False
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except ClientError as e:
            print(f"Error deleting file from S3: {e}")
            return False
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except ClientError:
            return False


# Global S3 service instance
s3_service = S3Service()

