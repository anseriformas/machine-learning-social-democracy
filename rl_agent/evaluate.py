import sys
# pyrefly: ignore [missing-import]
from sb3_contrib import MaskablePPO
# pyrefly: ignore [missing-import]
from sb3_contrib.common.wrappers import ActionMasker
from env import SocialDemocracyEnv

def mask_fn(env: SocialDemocracyEnv):
    return env.action_masks()

def main():
    model_path = "ppo_dynamic_social_democracy.zip"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print(f"Loading model from {model_path}...")
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Initializing environment...")
    base_env = SocialDemocracyEnv(headless=True)
    env = ActionMasker(base_env, mask_fn)
    
    num_episodes = 1
    
    for ep in range(num_episodes):
        print(f"Starting Episode {ep + 1}")
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            step += 1
            # Get available actions for logging
            choices = base_env.page.evaluate("dendryUI.dendryEngine.getCurrentChoices()")
            action_masks = env.action_masks()
            
            # Predict best action
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            scene_id = base_env.all_scenes[action]
            
            choice_text = "Unknown"
            if choices:
                for c in choices:
                    if c and c.get('id') == scene_id:
                        choice_text = c.get('title', [scene_id])[0]
                        break
            
            print(f"Step {step}: Agent chose '{choice_text}' (ID: {scene_id})")
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        print(f"Episode {ep + 1} finished with Total Reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    main()
