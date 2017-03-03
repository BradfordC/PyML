import Network
import Agent
import LineProblem
import Population

problem = LineProblem.LineProblem(.1, .01)
baseNet = Network.Network(LineProblem.Inputs, LineProblem.Outputs, [3])
agent = Agent.Agent(baseNet)

print(problem.GetFitness(agent, 100, .1))