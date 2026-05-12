import os
import sys
import gc
import gymnasium as gym
import numpy as np
from playwright.sync_api import sync_playwright
import fast_env

OBS_SIZE = 40  # must match fast_env.c

# simple coloured log so debug output is easy to scan
def _log(tag, msg, colour=None):
    codes = {"g": "\033[32m", "y": "\033[33m", "r": "\033[31m", "b": "\033[34m"}
    reset = "\033[0m"
    prefix = f"{codes.get(colour,'')}{tag}{reset}" if colour else tag
    print(f"{prefix} {msg}", flush=True)


class SocialDemocracyEnv(gym.Env):
    def __init__(self, headless=True, debug=False):
        super().__init__()
        self.headless = headless
        self.debug    = debug

        self.playwright = sync_playwright().start()
        self.browser    = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-gpu-compositing",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--disable-extensions",
                "--mute-audio",
            ],
        )

        # one persistent context + page — never torn down between episodes
        self.context = self.browser.new_context(ignore_https_errors=True)
        self.context.route("**/*.{mp3,ogg,wav,mp4,webm}", lambda r: r.abort())
        self.page = self.context.new_page()
        self.page.on("console",   lambda _: None)
        self.page.on("pageerror", lambda _: None)

        _log("[env]", f"loading game (headless={self.headless})", "b")
        self.page.goto(
            "file:///home/caleb/dynamic_social_democracy/out/html/index.html",
            wait_until="networkidle",
        )
        self.page.wait_for_selector("#content")
        self.page.wait_for_timeout(300)

        # kill all CSS animations — makes rendering instant
        self.page.add_style_tag(content=(
            "*, *::before, *::after {"
            "  animation-duration: 0s !important;"
            "  transition-duration: 0s !important; }"
        ))

        scenes = self.page.evaluate("Object.keys(dendryUI.game.scenes)")
        self.all_scenes  = list(sorted(scenes))
        self.num_actions = len(self.all_scenes)

        self.action_space      = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )

        self.prev_summary  = None
        self.episode_count = 0
        self._step_count   = 0

        _log("[env]", f"ready — {self.num_actions} scenes, obs size {OBS_SIZE}", "g")

    # ── dendry helpers ────────────────────────────────────────────────────────

    def _choose_by_id(self, target_id):
        choices = self.page.evaluate("dendryUI.dendryEngine.getCurrentChoices()") or []
        norm    = target_id if target_id.startswith("@") else "@" + target_id
        for i, c in enumerate(choices):
            if c and c.get("id") in {target_id, norm}:
                self.page.evaluate(f"dendryUI.dendryEngine.choose({i})")
                self._wait_for_choices()
                return True
        return False

    def _go_to_scene(self, scene_id):
        self.page.evaluate(f"dendryUI.dendryEngine.goToScene('{scene_id}')")
        self._wait_for_choices(timeout_ms=2000)

    def _wait_for_choices(self, timeout_ms=3000):
        try:
            self.page.wait_for_function(
                "() => { const c = dendryUI.dendryEngine.getCurrentChoices(); return c && c.length > 0; }",
                timeout=timeout_ms,
            )
        except Exception:
            pass  # timed out — probably a terminal state with no choices

    # ── state helpers ─────────────────────────────────────────────────────────

    def _get_qualities(self):
        return self.page.evaluate("dendryUI.dendryEngine.state.qualities") or {}

    def _summarize(self, q):
        if not q:
            return {}
        return {
            "year":                q.get("year"),
            "month":               q.get("month"),
            "resources":           q.get("resources"),
            "budget":              q.get("budget"),
            "dissent_percent":     q.get("dissent_percent"),
            "pro_republic":        q.get("pro_republic"),
            "president":           q.get("president"),
            "chancellor":          q.get("chancellor"),
            "chancellor_party":    q.get("chancellor_party"),
            "hindenburg_relation": q.get("hindenburg_relation"),
            "hindenburg_angry":    q.get("hindenburg_angry"),
            "coalition_dissent":   q.get("coalition_dissent"),
            "prussia_leader":      q.get("prussia_leader"),
            "rubicon":             q.get("rubicon", 0),
            "spd_seats":           q.get("spd_seats", 0),
            "nsdap_seats":         q.get("nsdap_seats", 0),
            "z_seats":           q.get("z_seats", 0),
            "ddp_seats":         q.get("ddp_seats", 0),
            "dnvp_seats":        q.get("dnvp_seats", 0),
            "dvp_seats":         q.get("dvp_seats", 0),
            "kpd_seats":         q.get("kpd_seats", 0),
            "game_over":           q.get("game_over", 0),
            "total_defeat":        q.get("total_defeat", 0),
            "ai_party_support": {
                "SPD":   q.get("spd_votes"),
                "KPD":   q.get("kpd_votes"),
                "Z":     q.get("z_votes"),
                "DDP":   q.get("ddp_votes"),
                "DVP":   q.get("dvp_votes"),
                "DNVP":  q.get("dnvp_votes"),
                "NSDAP": q.get("nsdap_votes"),
                "Other": q.get("other_votes"),
            },
            "relations": {
                "Z":     q.get("z_relation"),
                "KPD":   q.get("kpd_relation"),
                "DDP":   q.get("ddp_relation"),
                "DVP":   q.get("dvp_relation"),
                "DNVP":  q.get("dnvp_relation"),
                "NSDAP": q.get("nsdap_relation"),
                "LVP":   q.get("lvp_relation"),
            },
        }

    def _debug_state(self, q, reward=None, chosen_id=None):
        if not self.debug:
            return
        s   = self._summarize(q)
        sup = s.get("ai_party_support", {})
        rel = s.get("relations", {})
        colour = "g" if (reward or 0) >= 0 else "r"

        _log("[step]", (
            f"Episode = {self.episode_count} Step = {self._step_count} "
            f"{s.get('year')}-{str(s.get('month') or 0).zfill(2)}"
            + (f"  Reward = {reward:+.3f}" if reward is not None else "")
            + (f"  Agent Chose = {chosen_id}" if chosen_id else "")
        ), colour)
        _log("  Votes ", (
            f"SPD = {sup.get('SPD')}  NSDAP = {sup.get('NSDAP')}  KPD = {sup.get('KPD')}  "
            f"Z = {sup.get('Z')}  DDP = {sup.get('DDP')}  DVP = {sup.get('DVP')}  "
            f"DNVP = {sup.get('DNVP')}  Other = {sup.get('Other')}"
        ))
        _log("  Seats ", f"SPD = {s.get('spd_seats')}  NSDAP = {s.get('nsdap_seats') }  KPD = {s.get('kpd_seats')}  Z = {s.get('z_seats')}  DDP = {s.get('ddp_seats')}  DVP = {s.get('dvp_seats')}  DNVP = {s.get('dnvp_seats')}")
        _log("  Government   ", (
            f"Chancellor = {s.get('chancellor')} ({s.get('chancellor_party')})  " 
            f"Minister President = {s.get('prussia_leader')}  "
            f"President = {s.get('president')}  "
            f"Resources = {s.get('resources')}  Budget={s.get('budget')}"
        ))
        _log("  Health", (
            f"Dissent = {s.get('dissent_percent')}%  Pro-Republic = {s.get('pro_republic')}  "
            f"Coalition Dissent = {s.get('coalition_dissent')}  "
            f"Hindenburg Relation = {s.get('hindenburg_relation')}  Hindenburg Angry = {s.get('hindenburg_angry')}"
        ))
        _log("  Relations ", (
            f"KPD = {rel.get('KPD')}  Z = {rel.get('Z')}  DDP = {rel.get('DDP')}  "
            f"DVP = {rel.get('DVP')}  DNVP = {rel.get('DNVP')}  LVP = {rel.get('LVP')}  "
            f"NSDAP = {rel.get('NSDAP')}"
        ))
        _log("  Flags ", (
            f"Rubicon = {s.get('rubicon')}  Game Over = {s.get('game_over')}  "
            f"Total Defeat = {s.get('total_defeat')}"
        ))

    # ── action mask ───────────────────────────────────────────────────────────

    def action_masks(self):
        mask    = np.zeros(self.num_actions, dtype=bool)
        choices = self.page.evaluate("dendryUI.dendryEngine.getCurrentChoices()") or []

        def scene_of(c):
            if not c or "id" not in c:
                return None
            cid = c["id"]
            if isinstance(cid, str) and cid.startswith("@"):
                cid = cid[1:]
            return cid if cid in self.all_scenes else None

        valid   = [c for c in choices if c and c.get("canChoose") and scene_of(c)]
        decks   = [c for c in valid   if c.get("isDeck")]
        targets = decks if decks else valid

        for c in targets:
            mask[self.all_scenes.index(scene_of(c))] = True
        return mask

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self._step_count    = 0

        if self.debug:
            _log("[reset]", f"starting episode {self.episode_count}", "b")

        # fast reset via dendry — no browser reload
        try:
            self._go_to_scene("root")
            self.page.wait_for_timeout(100)
        except Exception:
            _log("[reset]", "engine broken, reloading page (slow fallback)", "y")
            self.page.reload(wait_until="networkidle")
            self.page.wait_for_selector("#content")
            self.page.wait_for_timeout(300)

        for scene in ["modinfo", "modinfo.infotext", "modinfo.flavors",
                      "root.start", "root", "root.1928_mod_mode"]:
            self._choose_by_id(scene)

        self.page.wait_for_timeout(100)

        q                 = self._get_qualities()
        self.prev_summary = self._summarize(q)
        obs               = fast_env.get_obs(q, OBS_SIZE).astype(np.float32)

        if self.debug:
            self._debug_state(q)

        return obs, {}

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        self._step_count += 1
        target_id = self.all_scenes[action]
        choices   = self.page.evaluate("dendryUI.dendryEngine.getCurrentChoices()") or []

        choice_idx = -1
        chosen_id  = None

        for i, c in enumerate(choices):
            if c and c.get("id") == target_id and c.get("canChoose"):
                choice_idx = i
                chosen_id  = c.get("id")
                break

        if choice_idx == -1:
            for i, c in enumerate(choices):
                if c and c.get("canChoose") and c.get("isDeck"):
                    choice_idx = i; chosen_id = c.get("id"); break
        if choice_idx == -1:
            for i, c in enumerate(choices):
                if c and c.get("canChoose"):
                    choice_idx = i; chosen_id = c.get("id"); break
        if choice_idx == -1 and choices:
            choice_idx = 0
            chosen_id  = (choices[0] or {}).get("id")

        if choice_idx != -1:
            self.page.evaluate(f"dendryUI.dendryEngine.choose({choice_idx})")
            self.page.wait_for_timeout(30)

        q       = self._get_qualities()
        obs     = fast_env.get_obs(q, OBS_SIZE).astype(np.float32)
        summary = self._summarize(q)

        # terminal detection
        chancellor       = q.get("chancellor", "")
        chancellor_party = q.get("chancellor_party", "")
        president        = q.get("president", "")
        fascist_win = (
            chancellor in {"Hitler", "Goebbels", "Göring", "Goering"}
            or chancellor_party == "NSDAP"
            or president in {"Hitler", "Göring", "Goering", "Frick"}
        )
        total_defeat   = q.get("total_defeat", 0) == 1
        game_over_flag = q.get("game_over", 0) == 1
        no_choices     = len(self.page.evaluate("dendryUI.dendryEngine.getCurrentChoices()") or []) == 0

        done            = False
        terminal_reward = None

        if fascist_win or total_defeat:
            terminal_reward = -100.0
            done = True
            if self.debug:
                _log("[step]", f"LOSS — fascist win ({chancellor or president})", "r")
        elif game_over_flag or no_choices:
            terminal_reward = 100.0
            done = True
            if self.debug:
                _log("[step]", "WIN — non-fascist ending", "g")

        reward = terminal_reward if terminal_reward is not None else \
                 float(fast_env.compute_reward(self.prev_summary, summary))

        if self.debug:
            self._debug_state(q, reward=reward, chosen_id=chosen_id)

        self.prev_summary = summary
        return obs, reward, done, False, {
            "chosen_id": chosen_id,
            "target_id": target_id,
            "n_choices": len(choices),
        }

    # ── cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        for attr in ("page", "context", "browser", "playwright"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                obj.close() if attr != "playwright" else obj.stop()
            except Exception:
                pass
            setattr(self, attr, None)
        gc.collect()