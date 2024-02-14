import random

class replayMemory():
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.push_count = 0

  def add_experience(self, experience):
    if len(self.memory) < self.capacity:
      self.memory.append(experience)
    else:
      self.memory[self.push_count % self.capacity] = experience
    self.push_count += 1

  def sample_batch(self, batch_size):
    if len(self.memory) >= batch_size:
      return random.sample(self.memory, batch_size)
    return