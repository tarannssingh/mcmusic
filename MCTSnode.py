import math

class MCTSnode:
    def __init__(self, features = None, genre = None, parent = None):
            
        self.features = features # audio featire dictionaru
        self.genre = genre # genre hypothesis like blues, jazz, pop
        self.parent = parent # reference to parent node

        self.children = [] # list of chile nodes from parent node
        self.exploredGenres = set() # set of genres already explored from this node (Self)

        self.visits = 0 # number of times this node has been visited 
        self.totalReward = 0.0 # sum of rewards from all simulations used in the UCB1 formula to balance exploration and exploitation

    def addChildren (self, genre):

        child = MCTSnode(features=self.features, genre=genre, parent=self)
        self.children.append(child)
        self.exploredGenres(genre)

        return child
    
    def UCBScore (self, exploredWeight = 1.0):

        if self.visits == 0: # edge case, ensures that if current node hasnt been visited, it gets visited
            return float('inf')
        elif self.parent is None or self.parent.visits == 0: # another edge case shouldnt happen but needed to makes sure parent node is visited
            return float('inf')
        else:
            exploitation = self.totalReward / self.visits # avg reward this node recieved; higher reward = better
            exploration = exploredWeight * math.sqrt(math.log(self.parent.visits) / self.visits) # encourages exploration of less visited nodes, if node is visited a lot this decreases, as parent node is visited more this increases
        
        return exploitation + exploration
    
    def isLeaf(self):
        
        # we know something is a leaf if there is no children
        # returns true or false
        return len(self.children) == 0
    
    def allPossibleExploredGenresFromChild (self, allGenres = None):
        
        if allGenres is None: # handeling is allGenres is not given

            if self.genre is not None: # if self.genre already represents a specific genre = True, fully explanded
                return True
            else: # if false, then it checks the list of explored genres if it is equal to 5 we know that it is fully expanded, we know this bc a set cannot contain duplicates
                return len(self.exploredGenres) >= 5
        else: # this is for is we have a list of all genres we want to use 
            return len(self.exploredGenres) >= len(allGenres)
    

    def mostVisitedChild (self):
        
        # if child hasnt been visited, it is definitely not the mostVisited
        if self.children is None:
            return None
        
        # initiate the variables going to be used
        # maxVisits is -1 b/c a child could be None or 0 but not -1
        mostVisits = None
        maxVisits = -1

        # for every child in the children list
        for child in self.children:
            # if the specific child visits is greater than maxVisits, reset maxVisits and mostVisits
            if child.visits > maxVisits:
                maxVisits = child.visits
                mostVisits = child
        return mostVisits
    

    def bestChild(self, exploarationWeight = 0.0):

        # best child according to ucb formula given the weight is 0

        # if children is none, there is no best child
        if self.children is None:
            return None
        
        # use avg reward for selection is explorationWeight is 0.0
        if exploarationWeight == 0.0:
            bestChild = None
            # setting best score to -inf b/c need to get a number not possible in set 
            bestScore = float('-inf')

            for child in self.children:
                # 1 to ensure no divide by 0
                visits = max(1, child.visits)
                score = child.totalReward / visits

                if score > bestScore:
                    bestScore = score
                    bestChild = child
            return bestChild
        
        #if explorationWeight is not 0, use the UCB scores from above
        else:
            bestChild = None
            bestScore = float('-inf')

            for child in self.children:
                score = child.UCBscore(exploarationWeight)

                if score > bestScore:
                    bestScore = score
                    bestChild = child
        return bestChild
    
    def averageReward(self):
        # gets average reward for use of explorationWeight = 0
        if self.visits == 0:
            return 0.0
        return self.totalReward / self.visits
    
    def __repr__(self):
        avgReward = self.averageReward()
        genreStr = self.genre if self.genre is not None else 'parent Root'
        return f"MCTSnode(genre = {genreStr}, visits={self.visits}, avgReward={avgReward:.4f})"