class ActionSpace:
    LEARN_MAC     = 0  
    EVICT_ENTRY   = 1  
    FLOOD         = 2  
    ASSIGN_MAC    = 3  
    BLOCK_PORT    = 4  
    UPDATE_CONFIG = 5  

    @staticmethod
    def get_all_actions():
        return [0, 1, 2, 3, 4, 5]
    
    