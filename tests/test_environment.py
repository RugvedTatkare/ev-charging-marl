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
        self.assertEqual(len(info["queue_lengths"]), 3)


    def test_ev_travels_on_road_segments(self):
        env = EVChargingEnvironment(num_evs=4, seed=19, max_steps=10)
        ev = env.evs[0]
        ev.battery = 20.0
        self.assertTrue(
            any(abs(ev.x - x) < 1e-8 for x in env.road_x)
            or any(abs(ev.y - y) < 1e-8 for y in env.road_y)
        )

        env.step({0: 0})
        # Grid roads are horizontal or vertical; the EV must remain on one.
        self.assertTrue(
            any(abs(ev.x - x) < 1e-8 for x in env.road_x)
            or any(abs(ev.y - y) < 1e-8 for y in env.road_y)
        )

    def test_done_after_max_steps(self):
        env = EVChargingEnvironment(num_evs=4, seed=11, max_steps=2)
        _, _, done_first, _ = env.step({})
        _, _, done_second, _ = env.step({})
        self.assertFalse(done_first)
        self.assertTrue(done_second)


if __name__ == "__main__":
    unittest.main()
