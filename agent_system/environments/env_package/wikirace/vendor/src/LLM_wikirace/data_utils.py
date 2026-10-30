import gzip
import json
import os
from pathlib import Path
from typing import Union, Optional
import numpy as np

import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from LLM_wikirace.config import DATA_DIR


class WikiGraph:
    def __init__(self, row_ptr: np.ndarray, col_indices: np.ndarray, id_to_title: dict, page_len: Optional[dict] = None, page_id_to_categories: Optional[dict] = None):
        '''
        Initializes the WikiGraph with the given row pointers, column indices, and ID-to-title mapping.
        Args:
            row_ptr (np.ndarray): Row pointers for the CSR representation of the graph.
            col_indices (np.ndarray): Column indices for the CSR representation of the graph.
            id_to_title (dict): Mapping from node IDs to page titles.
        '''
        self.row_ptr = row_ptr
        self.col_indices = col_indices
        self.matrix = csr_matrix((np.ones(len(col_indices)), col_indices, row_ptr), shape=(len(row_ptr) - 1, len(row_ptr) - 1))
        self.matrix_transpose = self.matrix.transpose(copy=True).tocsr()
        self.id_to_title = id_to_title
        
        self.page_len = page_len
        self.page_id_to_categories = page_id_to_categories

    def get_children(self, node_id: int):
        '''
        Returns the children of a given node in the graph.
        Args:
            node_id (int): The ID of the node for which to get children.
        Returns:
            np.ndarray: A numpy array of unique neighbor node IDs.
        '''
        children = self.matrix.getrow(node_id).nonzero()[1]
        return np.unique(children)
        

    def get_ancestors(self, node_id: int):
        '''
        Returns the ancestors of a given node in the graph.
        Args:
            node_id (int): The ID of the node for which to get ancestors.
        Returns:
            list: A list of ancestors node IDs.
        '''
        return self.matrix_transpose.getrow(node_id).nonzero()[1]

    def get_node_title(self, node: int):
        '''
        Gets the page title of a page (node in the wikigraph) by its ID.
        '''
        return self.id_to_title.get(str(node), None)
    
    def get_random_node(
        self, 
        rng: np.random.RandomState=None, 
        min_links: int = 0, 
        min_ancestors: int = 0,
        min_len: int= 0,
        forbidden_category_keywords: list=[], # e.g. ["'all_disambiguation_pages'", "'all_stub_articles'"]
        verbose: bool=True
    ):
        '''
        Returns a random node ID from the graph.
        Args:
            rng (np.random.RandomState, optional): A random number generator instance. If None, uses np.random.
            min_links (int, optional): Minimum number of links a node must have to be considered. Defaults to 0.
            min_ancestors (int, optional): Minimum number of incoming links node must have to be considered. Defaults to 0.
            min_len (int, optional): Minimum len of page in bytes to be considered. Defaults to 0. If not 0 and page_len was not loaded, throws exception.
            forbidden_category_keywords (list, optional): List of keywords that cannot appear in page categories for it to be considered. Defaults to empty list. If not empty and page_id_to_categories was not loaded, throws exception.
            verbose (bool, optional): Print a debug message containing info about selected page.
        Returns:
            int: A random node ID from the graph.
        '''
        
        if min_len != 0:
            assert self.page_len is not None, "Need to load page_len to filter by length."
        
        if forbidden_category_keywords != []:
            assert self.page_id_to_categories is not None, "Need to load categories to filter by categories."
            
        feasible_page_found = False
        while not feasible_page_found:        
            if rng is not None:
                random_key = rng.choice(list(self.id_to_title.keys()))
            else:
                random_key = np.random.choice(list(self.id_to_title.keys()))
                
            min_link_condition_met = self.get_children(int(random_key)).size >= min_links 
            min_ancestors_condition_met = self.get_ancestors(int(random_key)).size >= min_ancestors

            min_len_condition_met = self.page_len[str(random_key)] >= min_len if self.page_len is not None else True
            category_condition_met = (
                not(any( any(kw.lower() in category.lower() for kw in forbidden_category_keywords) for category in self.page_id_to_categories[str(random_key)] ))
                if self.page_id_to_categories is not None
                else True
            )

            if (
                min_link_condition_met and min_ancestors_condition_met
                and min_len_condition_met and category_condition_met
            ):
                if verbose:
                    print(
                        f"Page: {self.get_node_title(int(random_key))}| " 
                        + f"neighbors: {len(self.get_children(int(random_key)))}| " 
                        + f"ancestors: {len(self.get_ancestors(int(random_key)))}| " 
                        + (f"len: {self.page_len[str(random_key)]}| " if self.page_len is not None else "")
                        + (f"categories: {','.join(self.page_id_to_categories[str(random_key)])}" if self.page_id_to_categories is not None else "")
                        + "\n-----------------------------------------------------------------\n"
                    )
                feasible_page_found = True
        
        return int(random_key)

    def get_shortest_path_arrays(self, node1: int, node2: Union[int, None]=None):
        '''
        Computes the shortest path from node2 to node1 (or from all nodes if node2 is None).
        Args:
            node1 (int): The target node ID.
            node2 (int, optional): The starting node ID. If None, computes paths from all nodes.
        Returns:
            dict: A dictionary containing the distance matrix and predecessors array.
        '''
        dist_matrix, predecessors = shortest_path(
            self.matrix_transpose,
            indices=[node1, node2] if node2 is not None else [node1],
            return_predecessors=True,
            unweighted=True
        )
        
        return {
            "dist_matrix": dist_matrix,
            "predecessors": predecessors
        }
        
    def extract_path(self, predecessors: np.ndarray, start_node: int) -> list:
        '''
        Extracts the path from the predecessors array starting from the given start_node.
        Args:
            predecessors (np.ndarray): The array of predecessors for each node.
            start_node (int): The starting node ID from which to extract the path.
        Returns:
            list: A list of node IDs representing the path from start_node to the root node.
        '''

        assert predecessors.shape[0] == 1,\
            f"Predecessors array dim 0 should be 1, but got {predecessors.shape}, ensure get_shortest_path_arrays is called with node2=None"

        path = []
        current_node = start_node
        while current_node != -9999:  # -9999 indicates no predecessor
            path.append(current_node)
            current_node = predecessors[:, current_node].item()

        return path


def load_wikigraph(datapath: str = None) -> WikiGraph:
    '''
    Loads the Wikipedia graph from the specified directory.
    Args:
        datapath (str): Path to the directory containing the wiki graph data files.
    Returns:
        WikiGraph: An instance of the WikiGraph class containing the graph data.
    '''
    if datapath is None:
        datapath = os.path.join(DATA_DIR, "wikidata")

    graph_sparse = np.load(os.path.join(datapath, 'wiki_graph_sparse_filtered.npz'))
    row_ptr = graph_sparse['row_ptr']
    col_indices = graph_sparse['col_indices']
    
    with gzip.open(os.path.join(datapath, 'id_to_title_filtered.json.gz'), 'rt', encoding="utf-8") as f:
        id_to_title = json.load(f)
        
    assert len(id_to_title) == 549232, "You are using an older version of the wikigraph."
    
    try:
        with gzip.open(os.path.join(datapath, 'page_len.json.gz'), 'rt', encoding="utf-8") as f:
            page_len = json.load(f)
    except FileNotFoundError:
        page_len = None
    
    try:    
        with gzip.open(os.path.join(datapath, 'page_id_to_categories.json.gz'), 'rt', encoding="utf-8") as f:
            page_id_to_categories = json.load(f)
    except FileNotFoundError:
        page_id_to_categories = None
    
    return WikiGraph(row_ptr, col_indices, id_to_title, page_len, page_id_to_categories)


def load_test_pages(test_pages_path: Path)-> list[tuple[int,int]]:
    '''
    Loads the test pages from the specified path. Test pages expect a pandas dataframe with two columns in it for source and target pages...

    Args:
        test_pages_path (Path): Path to the file containing the test pages.
    Returns:
        list[tuple[int, int]]: A list of tuples, each containing a start page and a target page.
    '''

    # Load the test pages:
    df = pd.read_csv(test_pages_path)

    # convert to numpy array:
    return df.to_numpy().tolist()
    