
import os
from dotenv import load_dotenv
import pandas as pd
import shutil

class FileMetadataClassifier:
    def __init__(self) -> None:
        pass

    def collect_file_metadata(self, directory):
        files_data = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_info = os.stat(file_path)
                files_data.append({
                    'name': file,
                    'path': file_path,
                    'size': file_info.st_size,
                    'creation_date': file_info.st_ctime,
                    'modification_date': file_info.st_mtime,
                    'extension': os.path.splitext(file)[1]
                })
        files_data = pd.DataFrame(files_data)
        return files_data

    def categorize_files(self, metadata_df):
        categories = {
            'source_code': ['.py', '.java', '.cpp', '.ipynb'],
            'documents': ['.txt', '.docx', '.pdf'],
            'images': ['.jpg', '.png', '.gif']
        }
        
        def get_category(extension):
            for category, exts in categories.items():
                if extension in exts:
                    return category
            return 'others'

        metadata_df['category'] = metadata_df['extension'].apply(get_category)
        return metadata_df
    
    def organize_files(self, categorized_df, base_directory, move:bool=False):
        if not os.path.exists(base_directory):
            os.makedirs(base_directory, exist_ok=True)
        for category in categorized_df['prediction'].unique():
            category_path = os.path.join(base_directory, category)
            os.makedirs(category_path, exist_ok=True)
            
            for _, row in categorized_df[categorized_df['prediction'] == category].iterrows():
                file_name = row['path'].split("\\")[-1]
                if move:
                    shutil.move(row['path'], os.path.join(category_path, file_name))
                else:
                    shutil.copy(row['path'], os.path.join(category_path, file_name))

if __name__=="__main__":
    filemetadataclassifier=FileMetadataClassifier()
    # metadata = filemetadataclassifier.collect_file_metadata('data/')
    # print(filemetadataclassifier.categorize_files(metadata))
    expected_df = pd.read_csv("expected_df.tsv", sep='\t')
    filemetadataclassifier.organize_files(expected_df, 'organized')