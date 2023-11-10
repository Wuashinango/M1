from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from mesa import Agent, Model, time, space, DataCollector
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


class DirtyCell(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)

    def step(self):
        pass


class CleanerAgent(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)

    def move(self):
        possible_positions = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )

        new_position = self.random.choice(possible_positions)
        self.model.grid.move_agent(self, new_position)

    def clean_cell(self, dirtyCell):
        self.model.grid.remove_agent(dirtyCell)
        self.model.schedule.remove(dirtyCell)
        self.model.numDirtyCells -= 1

    def is_dirty(self,cellmates):
        for cellmate in cellmates:
            if isinstance(cellmate, DirtyCell):
                return cellmate
            
        return None

    def step(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        
        dirtyCell = self.is_dirty(cellmates)

        if dirtyCell:
            self.clean_cell(dirtyCell)
        else:
            self.move()
            self.model.totalAgentMovements += 1


class CleanerModel(Model):
    def __init__(self, M, N, numAgents, pCells, timeSteps) -> None:
        self.totalAgentMovements = 0
        self.M = M
        self.N = N
        self.grid = space.MultiGrid(self.M, self.N, True)
        self.numAgents = numAgents
        self.pCells = pCells
        self.pCleanCells = 1 - self.pCells
        self.timeSteps = timeSteps
        self.datacollector = DataCollector(model_reporters={'AgentMovements': 'totalAgentMovements','pCleanCells':'pCleanCells'})

        self.schedule = time.RandomActivation(self)

        self.numDirtyCells = int(self.M * self.N * self.pCells)
        self.running = self.numDirtyCells > 0

        # Dirty cells placed in random positions
        for i in range(self.numDirtyCells):
            dirtyCellPlaced = False

            while not dirtyCellPlaced:
                x = self.random.randrange(self.M)
                y = self.random.randrange(self.N)
                if self.grid.is_cell_empty((x, y)):
                    dirtyCellPlaced = True

            cell = DirtyCell(i, self)
            self.schedule.add(cell)
            self.grid.place_agent(cell, (x, y))

        # Cleaner robots placed in (1, 1)
        for i in range(self.numAgents):
            agent = CleanerAgent(i+self.numDirtyCells, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (1, 1))

    def step(self):
        self.pCleanCells = ((self.N * self.M) - self.numDirtyCells) / (self.N * self.M)
        self.running = self.numDirtyCells > 0

        # Contar el número de agentes DirtyCell
        self.numDirtyCells = sum(1 for agent in self.schedule.agents if isinstance(agent, DirtyCell))

        self.datacollector.collect(self)
        self.schedule.step()

def agent_portrayal(agent):
    portrayal = {"Shape": "rect",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "white",
                 "w": 1,
                 "h": 1}

    if isinstance(agent, DirtyCell):
        portrayal["Color"] = "brown"
    elif isinstance(agent, CleanerAgent):
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 1
    # Agregar texto con el número de DirtyCells
    if isinstance(agent, CleanerModel):
        portrayal["text"] = str(agent.numDirtyCells)

    return portrayal

ancho = 12  # Ancho de la cuadrícula
alto = 12   # Alto de la cuadrícula
grid = CanvasGrid(agent_portrayal, ancho, alto, 500, 500)
server = ModularServer(CleanerModel,
                       [grid],
                       "Cleaner Model",
                       {"M": ancho, "N": alto, "numAgents": 3, "pCells": 0.2, "timeSteps": 100})
server.port = 8521  # Puerto por defecto
server.launch()
