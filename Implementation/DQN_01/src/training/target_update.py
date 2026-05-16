from src.config import TARGET_UPDATE_N


class TargetUpdater:
    """
    Manages hard update of target network.
    Copies prediction network weights to target
    every TARGET_UPDATE_N training steps.
    """

    def __init__(self, networks):
        """
        Parameters:
            networks : DQNNetworks instance
                       contains both prediction and target network
        """
        self.networks      = networks
        self.update_every  = TARGET_UPDATE_N
        self.steps_done    = 0
        self.update_count  = 0
        self.next_update   = TARGET_UPDATE_N

    # ─────────────────────────────────────────
    # STEP — call this every training step
    # ─────────────────────────────────────────
    def step(self):
        """
        Called every training step.
        Increments counter.
        Checks if hard update is due.
        Performs copy if yes.

        Returns:
            updated : bool → True if copy happened this step
        """
        self.steps_done += 1

        # check if update is due
        if self.steps_done % self.update_every == 0:
            self._do_hard_update()
            return True

        return False

    # ─────────────────────────────────────────
    # HARD UPDATE — copy weights
    # ─────────────────────────────────────────
    def _do_hard_update(self):
        """
        Copies all weights and biases from
        prediction network to target network.
        Logs the update.
        """
        self.networks.hard_update()
        self.update_count += 1
        self.next_update   = self.steps_done + self.update_every

        print(f"\n  ── Target Network Updated ──────────")
        print(f"     Update #{self.update_count}")
        print(f"     At training step : {self.steps_done}")
        print(f"     Next update at   : {self.next_update}")
        print(f"  ────────────────────────────────────\n")

    # ─────────────────────────────────────────
    # STATS — for logging
    # ─────────────────────────────────────────
    def stats(self):
        """
        Prints current target updater status.
        """
        steps_until_next = self.next_update - self.steps_done

        print("\n── Target Updater ──────────────────")
        print(f"  Steps Done      : {self.steps_done}")
        print(f"  Updates Done    : {self.update_count}")
        print(f"  Update Every    : {self.update_every} steps")
        print(f"  Next Update In  : {steps_until_next} steps")
        print("────────────────────────────────────\n")