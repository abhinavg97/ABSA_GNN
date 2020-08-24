from xml.etree import ElementTree as ET
import pandas as pd
import logging
from ..utils import TextProcessing


class GCNLoader():
    """
    Class for parsing data from files and storing dataframe
    """

    def __init__(self, file_path, dataset_name):

        assert (file_path is not None and dataset_name is not None),\
            "file_path and dataset_name should be specified"
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.text_processor = TextProcessing()

    def parse_sem_eval(self):
        """
        Parses sem eval dataset
        Args:
            file_name: file containing the sem eval dataset in XML format

        Returns:
            parsed_data: pandas dataframe for the parsed
            data containing labels and text
        """
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        data_row = []
        for review in root.findall('Review'):
            for sentences in review:
                for sentence in sentences:
                    temp_row = ['lorem ipsum', []]
                    text = sentence.find('text').text
                    temp_row[0] = self.text_processor.process_text(text)
                    for opinions in sentence.findall('Opinions'):
                        for opinion in opinions:
                            polarity = opinion.get('polarity')
                            if(polarity == 'positive'):
                                temp_row[1] += [1]
                            else:
                                temp_row[1] += [-1]
                    data_row += [temp_row]
        parsed_data = pd.DataFrame(data_row, columns=['text', 'label'])
        return parsed_data

    def parse_twitter(self):
        """
        Parses twitter dataset
        Args:
            file_name: file containing the twitter dataset

        Returns:
            parsed_data: pandas dataframe for the parsed data
            containing labels and text
        """
        count = 0
        data_row = []
        with open(self.file_path, "r") as file1:
            for line in file1:
                stripped_line = line.strip()
                if count % 3 == 0:
                    temp_row = ['lorem ipsum', []]
                    temp_row[0] = self.text_processor.process_text(stripped_line)
                elif count % 3 == 2:
                    temp_row[1] += [int(stripped_line)]
                    data_row += [temp_row]
                count += 1
        parsed_data = pd.DataFrame(data_row, columns=['text', 'label'])
        return parsed_data

    def get_dataframe(self):
        """
        Returns pandas dataframe
        """
        if(self.dataset_name == "Twitter"):
            return self.parse_twitter()
        elif(self.dataset_name == "SemEval"):
            return self.parse_sem_eval()
        else:
            logging.error(
                "{} dataset not yet supported".format(self.dataset_name))
