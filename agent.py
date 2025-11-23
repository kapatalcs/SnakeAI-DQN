import torch
import random
import numpy
from collections import deque
from gameAI import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNET, QTrainer
from analyzer import plot
import signal
import sys
import os
import collections

MAX_MEMORY = 200_000
BATCH_SIZE = 2048
LEARNING_RATE = 0.0001
TARGET_UPDATE_FREQUENCY = 10


class SnakeSolver:
    def __init__(self, game):
        self.game = game
        self.w = game.w
        self.h = game.h
        self.block_size = 20
        self.deltas = [
            Point(self.block_size, 0),   
            Point(-self.block_size, 0), 
            Point(0, -self.block_size), 
            Point(0, self.block_size)    
        ]

    def is_path_safe_future(self, path, body_set):
        if not path:
            return False
            
        path_len = len(path)
        virtual_body = body_set.copy()
        
        for point in path:
            virtual_body.add(point)
        
        current_snake = self.game.snake
        if path_len < len(current_snake):
            for i in range(1, path_len + 1):
                segment_to_remove = current_snake[-i]
                if segment_to_remove in virtual_body:
                    virtual_body.remove(segment_to_remove)

        target_point = path[-1]
        
        if self.can_reach_tail(target_point, virtual_body):
            return True
            
        space = self.flood_fill_count(target_point, virtual_body, limit=len(current_snake))
        if space > len(current_snake):
            return True
            
        return False

    def get_neighbors(self, head):
        neighbors = []
        for delta in self.deltas:
            new_x = head.x + delta.x
            new_y = head.y + delta.y
            if new_x >= 0 and new_x < self.w and new_y >= 0 and new_y < self.h:
                neighbors.append(Point(new_x, new_y))
        return neighbors

    def can_reach_tail(self, start_node, body_set_checker):
        tail = self.game.snake[-1] 
        queue = collections.deque([start_node])
        visited = set([start_node])
        
        while queue:
            curr = queue.popleft()
            if curr == tail:
                return True
            
            for neighbor in self.get_neighbors(curr):
                if neighbor not in visited:
                    if (neighbor in body_set_checker) and (neighbor != tail):
                        continue
                    
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def flood_fill_count(self, start_node, body_set, limit=300):
        queue = collections.deque([start_node])
        visited = set([start_node])
        count = 0
        while queue:
            curr = queue.popleft()
            count += 1
            if count >= limit: return count
            for neighbor in self.get_neighbors(curr):
                if neighbor not in visited and neighbor not in body_set:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return count

    def bfs_shortest_path(self, start, target, body_set):
        queue = collections.deque([[start]])
        visited = set([start])
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == target: return path
            for neighbor in self.get_neighbors(node):
                if neighbor not in visited and neighbor not in body_set:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        return None

    def get_next_move_coords(self):
        head = self.game.head
        food = self.game.food
        
        body_set = set(self.game.snake)
        if self.game.snake[-1] in body_set:
            body_set.remove(self.game.snake[-1])

        path_to_food = self.bfs_shortest_path(head, food, body_set)   
        if path_to_food:
            if self.is_path_safe_future(path_to_food, body_set):
                return path_to_food[1]

            if self.game.steps_without_food > 100:
                 space = self.flood_fill_count(path_to_food[1], body_set, limit=len(self.game.snake)*2)
                 if space > len(self.game.snake):
                     return path_to_food[1]

        tail = self.game.snake[-1]
        path_to_tail = self.bfs_shortest_path(head, tail, body_set)
        
        if path_to_tail and len(path_to_tail) > 1:
            return path_to_tail[1]

        possible_moves = []
        neighbors = self.get_neighbors(head)
        
        for move in neighbors:
            if move in body_set: continue

            space = self.flood_fill_count(move, body_set, limit=len(self.game.snake))
            temp_body = body_set.copy()
            temp_body.add(move)
            if self.game.snake[-1] in temp_body:
                temp_body.remove(self.game.snake[-1])
                
            reachable = self.can_reach_tail(move, temp_body)
            
            score = space
            if reachable:
                score += 5000 
            
            possible_moves.append({'point': move, 'score': score})
        
        if possible_moves:
            possible_moves.sort(key=lambda x: x['score'], reverse=True)
            return possible_moves[0]['point']
            
        return None


class Agent:
    def __init__(self):
        self.numberOfGames = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen= MAX_MEMORY)
        self.model = Linear_QNET(28,2048,3)
        self.target_model = Linear_QNET(28,2048,3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() 
        self.trainer = QTrainer(self.model, self.target_model, learningRate=LEARNING_RATE, gamma=self.gamma)
        self.record = 0
        self.plotScores = []
        self.plotMeanScores = []
        self.plotRecordScores = []
        self.totalScore = 0
        self.load_agent_state()
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE
    
    def is_path_safe(self, game, point, snake_len):
        if game.isCollision(point):
            return 0.0

        total_grid_size = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)
        target_count = min(snake_len * 3, total_grid_size * 0.7)
        
        queue = [point]
        visited = set([point])
        count = 0
        body_set = set(game.snake)
        
        max_iterations = int(target_count * 1.5)
        iterations = 0

        while queue and iterations < max_iterations:
            curr = queue.pop(0)
            count += 1
            iterations += 1
            
            if count >= target_count:
                return 1.0
                
            neighbors = [
                Point(curr.x + BLOCK_SIZE, curr.y),
                Point(curr.x - BLOCK_SIZE, curr.y),
                Point(curr.x, curr.y - BLOCK_SIZE),
                Point(curr.x, curr.y + BLOCK_SIZE)
            ]
            
            for n in neighbors:
                # Sınırlar ve Çarpışma Kontrolü
                if 0 <= n.x < game.w and 0 <= n.y < game.h:
                    if n not in body_set and n not in visited:
                        visited.add(n)
                        queue.append(n)
        return min(count / target_count, 1.0)

    def save_agent_state(self, file_name='agent_state.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'number_of_games': self.numberOfGames,
            'record_score': self.record,
            'total_score': self.totalScore,
            'plot_scores': self.plotScores,
            'plot_mean_scores': self.plotMeanScores,
            'plot_record_scores': self.plotRecordScores,
        }, file_name)
        print(f"Agent state saved to {file_name}") 

    def load_agent_state(self, file_name='agent_state.pth'):
        file_name = os.path.join('./model', file_name)
        
        if os.path.exists(file_name):
            try:
                checkpoint = torch.load(file_name)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.numberOfGames = checkpoint['number_of_games']
                self.record = checkpoint['record_score']
                self.totalScore = checkpoint.get('total_score', 0)
                self.plotScores = checkpoint.get('plot_scores', [])
                self.plotMeanScores = checkpoint.get('plot_mean_scores', [])
                self.plotRecordScores = checkpoint.get('plot_record_scores', [])
                
                self.model.eval()
                self.target_model.eval()
                
                print(f"Agent state yüklendi. Oyun {self.numberOfGames + 1}'den devam ediyor. (Rekor: {self.record})")
            
            except Exception as e:
                print(f"Agent state yüklenirken hata oluştu: {e}. Sıfırdan başlanıyor.")
                self.target_model.load_state_dict(self.model.state_dict())
        else:
            print(f"Kayit dosyasi ({file_name}) bulunamadi. Sıfırdan başlanıyor.")
            self.target_model.load_state_dict(self.model.state_dict())


    def getState(self, game):
        head = game.snake[0]

        pointLeft = Point(head.x - BLOCK_SIZE, head.y)
        pointRight = Point(head.x + BLOCK_SIZE, head.y)
        pointUp = Point(head.x, head.y - BLOCK_SIZE)
        pointDown = Point(head.x, head.y + BLOCK_SIZE)

        directionLeft = game.direction == Direction.LEFT
        directionRight = game.direction == Direction.RIGHT
        directionUp = game.direction == Direction.UP
        directionDown = game.direction == Direction.DOWN

        danger_straight = (directionRight and game.isCollision(pointRight)) or \
                          (directionLeft and game.isCollision(pointLeft)) or \
                          (directionUp and game.isCollision(pointUp)) or \
                          (directionDown and game.isCollision(pointDown))

        danger_right = (directionUp and game.isCollision(pointRight)) or \
                       (directionDown and game.isCollision(pointLeft)) or \
                       (directionLeft and game.isCollision(pointUp)) or \
                       (directionRight and game.isCollision(pointDown))

        danger_left = (directionDown and game.isCollision(pointRight)) or \
                      (directionUp and game.isCollision(pointLeft)) or \
                      (directionRight and game.isCollision(pointUp)) or \
                      (directionLeft and game.isCollision(pointDown))

        def body_at(p):
            return p in game.snake[1:]

        danger_straight_body = (directionRight and body_at(pointRight)) or \
                               (directionLeft and body_at(pointLeft)) or \
                               (directionUp and body_at(pointUp)) or \
                               (directionDown and body_at(pointDown))

        danger_right_body = (directionUp and body_at(pointRight)) or \
                            (directionDown and body_at(pointLeft)) or \
                            (directionLeft and body_at(pointUp)) or \
                            (directionRight and body_at(pointDown))

        danger_left_body = (directionDown and body_at(pointRight)) or \
                           (directionUp and body_at(pointLeft)) or \
                           (directionRight and body_at(pointUp)) or \
                           (directionLeft and body_at(pointDown))

        dir_l = directionLeft
        dir_r = directionRight
        dir_u = directionUp
        dir_d = directionDown

        food_left  = game.food.x < head.x
        food_right = game.food.x > head.x
        food_up    = game.food.y < head.y
        food_down  = game.food.y > head.y

        tail = game.snake[-1]
        second_last = game.snake[-2] if len(game.snake) > 1 else tail
        tail_dir_x = tail.x - second_last.x
        tail_dir_y = tail.y - second_last.y

        tail_left  = tail_dir_x < 0
        tail_right = tail_dir_x > 0
        tail_up    = tail_dir_y < 0
        tail_down  = tail_dir_y > 0

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        
        point_straight = [pointRight, pointDown, pointLeft, pointUp][idx]
        point_right_turn = [pointDown, pointLeft, pointUp, pointRight][idx]
        point_left_turn = [pointUp, pointRight, pointDown, pointLeft][idx]
        
        is_safe_straight = self.is_path_safe(game, point_straight, len(game.snake))
        is_safe_right = self.is_path_safe(game, point_right_turn, len(game.snake))
        is_safe_left = self.is_path_safe(game, point_left_turn, len(game.snake))

        tail_loc_left = tail.x < head.x
        tail_loc_right = tail.x > head.x
        tail_loc_up = tail.y < head.y
        tail_loc_down = tail.y > head.y

        def check_danger_ahead(direction, distance):
            if direction == Direction.RIGHT:
                point = Point(head.x + BLOCK_SIZE * distance, head.y)
            elif direction == Direction.LEFT:
                point = Point(head.x - BLOCK_SIZE * distance, head.y)
            elif direction == Direction.UP:
                point = Point(head.x, head.y - BLOCK_SIZE * distance)
            elif direction == Direction.DOWN:
                point = Point(head.x, head.y + BLOCK_SIZE * distance)
            return game.isCollision(point)
    
        danger_2_straight = check_danger_ahead(game.direction, 2)
        danger_3_straight = check_danger_ahead(game.direction, 3)
    
        snake_length_normalized = min(len(game.snake) / 500.0, 1.0)

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(danger_straight_body),
            int(danger_right_body),
            int(danger_left_body),

            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),

            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),

            int(tail_left),
            int(tail_right),
            int(tail_up),
            int(tail_down),

            int(is_safe_straight),
            int(is_safe_right),
            int(is_safe_left),

            int(tail_loc_left),
            int(tail_loc_right),
            int(tail_loc_up),
            int(tail_loc_down),

             int(danger_2_straight),
            int(danger_3_straight),
            snake_length_normalized
        ]

        return numpy.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            weights = []
            for exp in self.memory:
                reward = exp[2]
                if reward > 5:
                    weights.append(5.0)
                elif reward > 0:
                    weights.append(2.0)
                else:
                    weights.append(1.0)
            
            weights = numpy.array(weights)
            weights = weights / weights.sum()
            
            indices = numpy.random.choice(
                len(self.memory), 
                size=BATCH_SIZE, 
                p=weights
            )
            miniSample = [self.memory[i] for i in indices]
        else:
            miniSample = list(self.memory)
        
        states, actions, rewards, next_states, dones = zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, next_states, dones)


    def trainShortMemory(self, state, action, reward, next_state, done):
        self.trainer.trainStep(state, action, reward, next_state, done)

    def get_action_from_coord(self, game, next_point):
        if next_point is None:
            return [1, 0, 0]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        
        x = next_point.x - game.head.x
        y = next_point.y - game.head.y
        
        move_dir = None
        if x == BLOCK_SIZE: move_dir = Direction.RIGHT
        elif x == -BLOCK_SIZE: move_dir = Direction.LEFT
        elif y == BLOCK_SIZE: move_dir = Direction.DOWN
        elif y == -BLOCK_SIZE: move_dir = Direction.UP
        
        if move_dir == game.direction:
            return [1, 0, 0]
        elif move_dir == clock_wise[(idx + 1) % 4]:
            return [0, 1, 0]
        elif move_dir == clock_wise[(idx - 1) % 4]:
            return [0, 0, 1]
        else:
            return [1, 0, 0]

    def getAction(self, state, game):
        if game.score >= 150: 
            solver = SnakeSolver(game)
            next_coord = solver.get_next_move_coords()
            
            if next_coord:
                finalMove = self.get_action_from_coord(game, next_coord)
                return finalMove

        self.epsilon = 80 - self.numberOfGames
        if self.epsilon < 0: self.epsilon = 0
             
        finalMove = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            finalMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove

def train():
    agent = Agent()
    game = SnakeGameAI()

    def signal_handler(sig, frame):
        print("\nEğitim durduruldu. Agent state kaydediliyor...")
        agent.save_agent_state()
        print("Agent state başarıyla kaydedildi.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        while True:
            oldState = agent.getState(game)
            finalMove = agent.getAction(oldState, game)

            reward, done, score = game.play_step(finalMove)
            newState = agent.getState(game)

            agent.trainShortMemory(oldState, finalMove, reward, newState, done)
            agent.remember(oldState, finalMove, reward, newState, done)

            if done:
                game.reset()
                agent.numberOfGames += 1
                agent.trainLongMemory()

                if agent.numberOfGames % TARGET_UPDATE_FREQUENCY == 0: 
                    agent.target_model.load_state_dict(agent.model.state_dict())
                    print(f"Game {agent.numberOfGames}: Target network updated.")

                if agent.numberOfGames % 50 == 0:
                    agent.save_agent_state()

                if score > agent.record:
                    agent.record = score

                if score >= 300:
                    backup_name = f"model_record_{score}.pth"
                    agent.save_agent_state(file_name=backup_name)
                    print(f"!!! TEBRİKLER! YENİ BİR ZİRVE: {score}. Model '{backup_name}' olarak yedeklendi.")

                print(f"Game {agent.numberOfGames}, Score {score}, Record {agent.record}")

                agent.plotScores.append(score)
                agent.totalScore += score
                meanScore = agent.totalScore / agent.numberOfGames
                agent.plotMeanScores.append(meanScore)
                agent.plotRecordScores.append(agent.record)
                plot(agent.plotScores, agent.plotMeanScores, agent.plotRecordScores)
    finally:
        print("\n'Finally' bloğu çalıştı. Model son kez kaydediliyor...")
        agent.save_agent_state()
        print("Model başarıyla 'model/model.pth' olarak kaydedildi.")
if __name__ == '__main__':
    train()