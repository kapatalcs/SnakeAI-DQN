import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy 

pygame.init()
font = pygame.font.Font('arial.ttf',25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point','x, y')
BLOCK_SIZE = 20
SPEED = 0

class SnakeGameAI():
    def __init__(self,w = 640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()
        


    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(int(self.w/2),int(self.h/2))
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE,self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE),self.head.y),]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        self.steps_without_food = 0
        self.prev_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)


    def place_food(self):
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)
        if self.food in self.snake:
            self.place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        oldHead = self.head
        self.move(action)
        self.snake.insert(0, self.head)
        ate = False
        reward = 0
        game_over = False

        if self.head == self.food:
            self.score += 1
            reward = 10 + (len(self.snake) * 0.1)
            self.place_food()
            ate = True
            self.steps_without_food = 0
            self.prev_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        else:
            self.snake.pop()
            self.steps_without_food += 1

        if self.isCollision() or self.frame_iteration > 500 * len(self.snake):
            game_over = True
            reward = -50
            return reward, game_over, self.score

        if not ate:
            if len(self.snake) > 50:
                reward += 0.5
            if len(self.snake) > 100:
                reward += 1.0
            if len(self.snake) > 200:
                reward += 2.0

            oldDistance = abs(oldHead.x - self.food.x) + abs(oldHead.y - self.food.y)
            newDistance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            
            if newDistance < oldDistance:
                reward += 0.5
            reward -= 0.01

            def min_body_dist(pt, snake_list):
                if len(snake_list) <= 1:
                    return 9999
                d = min([abs(pt.x - s.x) + abs(pt.y - s.y) for s in snake_list[1:]])
                return d
            
            old_body_dist = min_body_dist(oldHead, self.snake) 
            new_body_dist = min_body_dist(self.head, self.snake)
            
            if new_body_dist < old_body_dist:
                reward -= 0.8

            if self.steps_without_food > 100:
                reward -= 2.0

        self.update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def isCollision(self, point = None):
        if point is None:
            point = self.head
        if point.x > self.w-BLOCK_SIZE or point.x < 0 or point.y > self.h-BLOCK_SIZE or point.y < 0:
            return True
        if point in self.snake[1:]:
            return True
        return False
    
    def move(self,action):
        clock_wise =[Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        indx = clock_wise.index(self.direction)
        newDirection = self.direction

        if numpy.array_equal(action, [1,0,0]):
            newDirection = clock_wise[indx]
        elif numpy.array_equal(action, [0,1,0]):
            newDirection = clock_wise[(indx+1)%4]
        elif numpy.array_equal(action, [0,0,1]):
            newDirection = clock_wise[(indx-1)%4]
        self.direction = newDirection
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x,y)

    def update_ui(self):
        self.display.fill((0,0,0))
        snake_len = len(self.snake)
        for i, point in enumerate(self.snake):
            if i == 0:
                color = (0, 0, 255)
                pygame.draw.rect(self.display, color, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
                eye_color = (200, 200, 255)
                pygame.draw.rect(self.display, eye_color, pygame.Rect(point.x + 5, point.y + 5, 4, 4))
                pygame.draw.rect(self.display, eye_color, pygame.Rect(point.x + 11, point.y + 5, 4, 4))

            else:
                max_lightness = 150
                progress_factor = i / snake_len

                r = int(progress_factor * max_lightness)
                g = int(progress_factor * max_lightness)
                b = 255
                color = (min(255, r), min(255, g), b)

                pygame.draw.rect(self.display, color, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display,(255,0,0), pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, (255,255,255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()