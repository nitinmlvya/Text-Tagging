import os
import pandas as pd

METADATA = r'metadata/mapping_conv_topic.train.txt'
LIST_FILES = r'tagging_test'
CORPUS_CSV = r'conversation_data.csv'

def map_file_ids_with_topics():
    ids_topics = {}
    with open(METADATA) as fp:
        text = [x.strip('\r\n') for x in fp.readlines()]
        text = [x.replace('"', '') for x in text]
        ids = [x.split()[0] for x in text]
        topics = [' '.join(x.split()[1:]) for x in text]
        ids_topics = dict(zip(ids, topics))
    print(f'Total metadata files: {len(ids_topics)}')
    return ids_topics


def conversation_with_topic(ids_topics):
    corpus = {'file_id': [], 'conversation': [], 'topic': []}
    for fname in os.listdir(LIST_FILES):
        file_id = fname.split('.')[1]
        if file_id in ['DS_Store']:
            continue
        topic = ids_topics[file_id]
        with open(f'{LIST_FILES}/{fname}') as fp:
            text = [x.strip('\r\n') for x in fp.readlines()]
            for t in text:
                corpus['file_id'].append(file_id)
                corpus['topic'].append(topic)
                corpus['conversation'].append(t)
    print(f'Total list of conversation sentences: {len(corpus["conversation"])}')
    return corpus

def convert_to_csv(corpus):
    # Convert to dataframe
    df = pd.DataFrame(corpus)
    df.to_csv(f'{CORPUS_CSV}', index=False)
    print(f'Conversations + topics are saved in {CORPUS_CSV}')

if __name__=='__main__':
    ids_topics = map_file_ids_with_topics()
    corpus = conversation_with_topic(ids_topics)
    convert_to_csv(corpus)
