from IPython import display
import matplotlib.pyplot as plt

plt.ion()

def plot(scores, meanScores, recordScores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label='Last Score', color='blue')
    plt.plot(meanScores, label='Mean Score', color='orange')
    plt.plot(recordScores, label='Record Score', color='green')

    plt.ylim(ymin=0)
    plt.legend(loc='upper left')

    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]), color='blue')
    if len(meanScores) > 0:
        plt.text(len(meanScores)-1, meanScores[-1], str(round(meanScores[-1],1)), color='orange')
    if len(recordScores) > 0:
        plt.text(len(recordScores)-1, recordScores[-1], str(recordScores[-1]), color='green')


