import unittest

from src.environment import EVChargingEnvironment


class EnvironmentSmokeTests(unittest.TestCase):
    def test_step_returns_expected_shapes(self):
        env = EVChargingEnvironment(num_evs=6, seed=7, max_steps=3)
        states = [env.observe(i) for i in range(6)]
        self.assertEqual(len(states), 6)

        next_states, rewards, done, info = env.step({})
        self.assertEqual(len(next_states), 6)
        self.assertEqual(len(rewards), 6)
        self.assertFalse(done)
        self.assertEqual(info["queue_lengths"].__len__(), 3)

    def test_done_after_max_steps(self):
        env = EVChargingEnvironment(num_evs=4, seed=11, max_steps=2)
        _, _, done_first, _ = env.step({})
        _, _, done_second, _ = env.step({})
        self.assertFalse(done_first)
        self.assertTrue(done_second)


if __name__ == "__main__":
    unittest.main()
