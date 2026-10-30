import os
from .game_engine import WikiRaceEngine, GameState
from .data_utils import load_wikigraph, load_test_pages
from .config import DATA_DIR


def get_possible_game_ids(difficulty: str) -> list[int]:
    """
    Returns a list of possible game IDs for the given difficulty level.

    Args:
        difficulty (str): The difficulty level ('easy', 'medium', 'hard').
    Returns:
        list[int]: A list of possible game IDs.
    """
    if difficulty == 'easy':
        return list(range(0, 200))
    elif difficulty == 'medium':
        return list(range(0, 150))
    elif difficulty == 'hard':
        return list(range(0, 100))
    else:
        raise ValueError(f"Invalid difficulty level: {difficulty}. Choose from 'easy', 'medium', 'hard'.")


class WikiRaceEnv():
    """Gym-style Environment for the WikiRace game."""

    def __init__(self, difficulty: str = 'easy', game_id: int = 0, seed: int = 42, max_choices: int = 50, max_path_length: int = 30, wikigraph_path: str = None, custom_page_id_path: str = None):
        assert difficulty in ['easy', 'medium', 'hard'], "Difficulty must be one of 'easy', 'medium', or 'hard'."
        self.difficulty = difficulty
        
        possible_game_ids = get_possible_game_ids(difficulty)
        assert game_id in possible_game_ids, f"Game ID must be an integer between {min(possible_game_ids)} and {max(possible_game_ids)} for difficulty '{difficulty}'."

        wikigraph = load_wikigraph(wikigraph_path)

        self.game_id = game_id
        self.current_state = None  
        self.game_engine = WikiRaceEngine(
            wikigraph,
            max_choices=max_choices,
            max_path_length=max_path_length,
            game_seed=seed
        )
        
        if custom_page_id_path is not None:
            test_pages = load_test_pages(custom_page_id_path)
        else:
            assert self.difficulty.lower() in ["easy", "medium", "hard"], "Difficulty must be one of easy, medium or hard"
            test_pages = load_test_pages(os.path.join(DATA_DIR, "splits", f"{self.difficulty.lower()}.csv"))
    
        self.start_page = test_pages[self.game_id][0]
        self.target_page = test_pages[self.game_id][1]        
    
    def reset(self) -> GameState:
        """Resets the environment to start a new game."""
        self.current_state = self.game_engine.initialize_game(
            inference_engine_name="custom",
            current_page=self.start_page,
            target_page=self.target_page,
            game_id=self.game_id
        )
        return self.current_state
    
    def step(self, action: int) -> GameState:
        """
        Takes an action in the environment and returns the new state, reward, done flag, and info.

        Args:
            action (int): The index of the chosen link from the available choices.
        Returns:
            new_state: The new game state after taking the action.
        """
        self.current_state = self.game_engine.step(self.current_state, action)
        
        return self.current_state
