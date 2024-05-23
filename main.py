from FileMetadataClassifier import *
from Summarizer import *

if __name__=="__main__":
    filemetadataclassifier=FileMetadataClassifier()
    metadata_df = filemetadataclassifier.collect_file_metadata('data/')
    txt_df = metadata_df[metadata_df['extension']=='.txt']
    summarizer = Summarizer()
    txt_df['summary'] = txt_df['path'].apply(lambda x: summarizer.summerize_text(x))
    print(txt_df)
    txt_df.to_csv('summary_df.tsv', index=False, sep='\t')

    expected_df = pd.read_csv("expected_df.tsv", sep='\t')
    filemetadataclassifier.organize_files(expected_df, 'organized')