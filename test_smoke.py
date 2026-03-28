import sys, traceback
sys.path.insert(0, '.')

from env.models import Action
from env.environment import CustomerSupportEnv
from env.grader import grade_episode

env = CustomerSupportEnv()
obs = env.reset(0)
a = Action.model_validate({'action_type': 'reply', 'message': 'Hello world this is a test response'})
r = env.step(a)
s = env.state()
print('history:', s.history)
print('state dict keys:', list(s.model_dump().keys()))

try:
    g = grade_episode(0, s.history, s.model_dump())
    print('grade OK:', g)
except Exception:
    traceback.print_exc()
