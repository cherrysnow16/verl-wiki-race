from typing import Optional
from typing_extensions import Self
import random
from dataclasses import dataclass
from LLM_wikirace.data_utils import WikiGraph


@dataclass
class GameState:
    """
    Represents the state of a WikiRace game at a particular point in time.
    
    Attributes:
        current_page_id (int): The ID of the current page.
        target_page_id (int): The ID of the target page to reach.
        current_page (str): The title of the current page.
        target_page (str): The title of the target page.
        available_choices (list[str]): List of available page titles to choose from.
        available_choices_ids (list[int]): List of page IDs corresponding to available choices.
        status (str): Current game status ("in progress", "completed", "failed").
        step_number (int): Current step number in the game.
        game_id (int): Unique identifier for this game.
        history (list[str]): History of pages visited in this game.
        fail_message (Optional[str]): Message describing why the game failed, if applicable.
        seed (int): Random seed for reproducibility.
    """
    
    inference_engine_name: str
    start_timestamp: str
    start_page: str
    start_page_id: int
    current_page_id: int
    target_page_id: int
    current_page: str
    target_page: str
    available_choices: list[str]
    available_choices_ids: list[int]
    status: str
    step_number: int
    game_id: int
    history: list[str]
    fail_message: Optional[str] = None
    initial_shortest_possible_path: list[str] = None
    
    def __post_init__(self):
        """Validate the game state after initialization."""
        if len(self.available_choices) != len(self.available_choices_ids):
            raise ValueError("available_choices and available_choices_ids must have the same length")
        
        assert self.status in ["in progress", "completed", "failed"], f"Invalid status: {self.status}"

    def is_completed(self) -> bool:
        """Check if the game is completed."""
        return self.status == "completed"
    
    def is_failed(self) -> bool:
        """Check if the game has failed."""
        return self.status == "failed"
    
    def is_in_progress(self) -> bool:
        """Check if the game is still in progress."""
        return self.status == "in progress"
    
    def get_choice_id(self, choice: str) -> Optional[int]:
        """Get the page ID for a given choice title."""
        try:
            choice_idx = self.available_choices.index(choice)
            return self.available_choices_ids[choice_idx]
        except ValueError:
            return None
    
    def add_to_history(self, page_title: str) -> None:
        """Add a page to the game history."""
        self.history.append(page_title)
    
    def get_path_length(self) -> int:
        """Get the current path length (number of steps taken)."""
        return len(self.history) - 1  # Subtract 1 because initial page is not a step

    def exception_handling(self, e: Exception) -> Self:
        """Handle an exception during the game."""
        
        # Update specific internal attributes to reflect the exception:
        self.status = "failed"
        self.fail_message = f"Game ID: {self.game_id} failed with exception: {e.__class__.__name__}: {e}"
        return self


class WikiRaceEngine:
    def __init__(
        self,
        link_graph: WikiGraph,
        max_choices:int=50,
        max_path_length:int=1000,
        min_links:int=0,
        verbose:bool=True,
        game_seed: int = 42
    ):
        '''
        Initializes the WikiRaceEngine with a link graph and game parameters.
        Args:
            link_graph (WikiGraph): The graph representing the Wikipedia pages and links.
            max_choices (int): Maximum number of choices available to the player at each step.
            max_path_length (int): Maximum length of the path allowed in the game.
        Returns:
            None
        '''
        self.link_graph = link_graph
        self.max_choices = max_choices
        self.max_path_length = max_path_length
        self.min_links = min_links
        self.verbose = verbose
        self.game_seed = game_seed
        self.rng = random.Random(game_seed)
    
    def initialize_game(self, 
                       inference_engine_name: str,
                       current_page: Optional[int] = None,
                       target_page: Optional[int] = None, 
                       game_id: int = 0,
                       start_timestamp: str = None,
                       ) -> GameState:
        """
        Initialize a new game and return the initial game state.
        
        Args:
            current_page (Optional[int]): Starting page ID. If None, randomly selected.
            target_page (Optional[int]): Target page ID. If None, randomly selected.
            game_id (int): Unique identifier for this game.
            seed (int): Random seed for reproducibility.
            
        Returns:
            GameState: Initial game state with available choices.
        """
        rng = self.rng
        
        # Randomly select pages if not provided
        if current_page is None:
            current_page = self.link_graph.get_random_node(rng=rng, min_links=self.min_links, verbose=self.verbose)
        if target_page is None:
            target_page = self.link_graph.get_random_node(rng=rng, min_links=self.min_links, verbose=self.verbose)
        
        # Ensure current and target pages are different
        while current_page == target_page:
            target_page = self.link_graph.get_random_node(rng=rng, min_links=self.min_links, verbose=self.verbose)
        
        # Get initial choices
        neighbors = list(self.link_graph.get_children(current_page))
        target_shortest_path_arrays = self.link_graph.get_shortest_path_arrays(target_page)
        
        available_choices, available_choices_ids = self._get_choices(neighbors=neighbors,
                                target_shortest_path_arrays=target_shortest_path_arrays,
                                rng=rng)
        
        initial_shortest_possible_path = [
            self.link_graph.get_node_title(node_id) for node_id 
            in self.link_graph.extract_path(
                target_shortest_path_arrays["predecessors"],
                current_page
            )
        ]

        return GameState(
            start_timestamp=start_timestamp,
            inference_engine_name=inference_engine_name,
            start_page=self.link_graph.get_node_title(current_page),
            start_page_id=current_page,
            current_page_id=current_page,
            target_page_id=target_page,
            current_page=self.link_graph.get_node_title(current_page),
            target_page=self.link_graph.get_node_title(target_page),
            available_choices=list(available_choices.keys()),
            available_choices_ids=available_choices_ids,
            status="in progress",
            step_number=0,
            game_id=game_id,
            history=[self.link_graph.get_node_title(current_page)],
            initial_shortest_possible_path=initial_shortest_possible_path
        )
    

    def step(self, game_state: GameState, choice: int, rng: Optional[random.Random] = None) -> GameState:
        """
        Takes a step in the game based on the player's choice.

        - Validate the choice is valid
        - Update the new optimal path
        - Update the game status
        - Provide the next set of available choices

        Args:
            game_state (GameState): Current game state.
            choice (str): The player's choice of the next page to navigate to.
            rng (Optional[random.Random]): Random number generator for reproducible choice selection.
            
        Returns:
            GameState: Updated game state after the step.
        """

        rng = self.rng
        
        choice = game_state.available_choices[choice]

        # Check if choice is valid or max path length reached
        if choice not in game_state.available_choices or game_state.step_number >= self.max_path_length:

            if choice not in game_state.available_choices:
                fail_message = f"Game ID: {game_state.game_id} failed, invalid choice: {choice} not in {game_state.available_choices}"
            elif game_state.step_number >= self.max_path_length:
                fail_message = f"Game ID: {game_state.game_id} failed, max path length reached: {game_state.step_number} >= {self.max_path_length}"

            # TODO: Investigate updating the game state rather than creating a new one...
            # Is the history ever being updated?
            return GameState(
                inference_engine_name=game_state.inference_engine_name,
                start_timestamp=game_state.start_timestamp,
                start_page=game_state.start_page,
                start_page_id=game_state.start_page_id,
                current_page_id=game_state.current_page_id,
                target_page_id=game_state.target_page_id,
                current_page=game_state.current_page,
                target_page=game_state.target_page,
                available_choices=game_state.available_choices,
                available_choices_ids=game_state.available_choices_ids,
                status="failed",
                step_number=game_state.step_number,
                game_id=game_state.game_id,
                history=game_state.history.copy(),
                fail_message=fail_message
            )
        
        # Get the new current page
        choice_idx = game_state.available_choices.index(choice)
        new_current_page_id = game_state.available_choices_ids[choice_idx]
        new_current_page = self.link_graph.get_node_title(new_current_page_id)
        
        # Update history
        new_history = game_state.history.copy()
        new_history.append(new_current_page)
        
        # Check if game is completed
        if new_current_page_id == game_state.target_page_id:
            status = "completed"
            new_available_choices = []
            new_available_choices_ids = []
        else:
            status = "in progress"
            # Get new choices for the next step
            neighbors = list(self.link_graph.get_children(new_current_page_id))
            target_shortest_path_arrays = self.link_graph.get_shortest_path_arrays(game_state.target_page_id)
            
            new_available_choices, new_available_choices_ids = self._get_choices(
                neighbors, target_shortest_path_arrays, rng
            )
            new_available_choices = list(new_available_choices.keys())
        
        return GameState(
            inference_engine_name=game_state.inference_engine_name,
            start_timestamp=game_state.start_timestamp,
            start_page=game_state.start_page,
            start_page_id=game_state.start_page_id,
            current_page_id=new_current_page_id,
            target_page_id=game_state.target_page_id,
            current_page=new_current_page,
            target_page=game_state.target_page,
            available_choices=new_available_choices,
            available_choices_ids=new_available_choices_ids,
            status=status,
            step_number=game_state.step_number + 1,
            game_id=game_state.game_id,
            history=new_history
        )
    
    def _get_choices(self, neighbors: list[int],
                        target_shortest_path_arrays: dict,
                        rng: random.Random
     )-> tuple[dict[str, int], list[int]]:
        '''
        Updates the available choices by including all top choices and filling rest with random sampling.

        Args:
            neighbors (list): List of new neighbors to consider for updating choices.
            current_page (int): The current page ID.
            target_shortest_path_arrays (dict): The shortest path arrays for the target page. Contains "predecessors" and "dist_matrix".
            rng (random.Random): The random number generator.
        '''
        available_choices = {}
        
        if len(neighbors) > self.max_choices:
            # Calculate how many top choices and random choices to include
            num_top_choices = self.max_choices
            
            # Get top choices sorted by shortest path distance
            # Using pre-computed distances (more efficient)
            top_neighbors = sorted(
                neighbors,
                key=lambda x: target_shortest_path_arrays["dist_matrix"][0, x],
                reverse=False
            )
            
            # Select the specified proportion of top choices
            final_neighbors = top_neighbors[:num_top_choices]
            
            # shuffle the neighbours
            rng.shuffle(final_neighbors)

            selected_neighbors = final_neighbors
        
        else:
            selected_neighbors = neighbors
        
         # Process the selected neighbors to get the available choices as text instead of just ids.
        for neighbor in selected_neighbors:
            available_choices[self.link_graph.get_node_title(neighbor)] = neighbor

        return available_choices, selected_neighbors