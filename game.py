import gymnasium as gym
import ale_py
import pygame
import sys
import time

# 初期化関数
def init_game():
    """ゲームと画面の初期化を行う関数"""
    # Pygameの初期化
    pygame.init()
    # MsPacmanの環境を作成
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    # 初期状態の取得
    observation, _ = env.reset()
    # 画面サイズの設定（元のサイズを2倍に拡大）
    original_height, original_width = observation.shape[0], observation.shape[1]
    scale_factor = 2  # 拡大倍率
    screen_width = original_width * scale_factor
    screen_height = original_height * scale_factor
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("MS Pacman")
    
    return env, screen

# 画面更新関数
def update_display(screen, observation):
    """観測結果を画面に表示する関数"""
    # NumPy配列からPygameのSurfaceに変換
    surf = pygame.surfarray.make_surface(observation.swapaxes(0, 1))
    # 画面サイズを取得
    screen_width, screen_height = screen.get_size()
    # 画像を画面サイズに合わせて拡大
    scaled_surf = pygame.transform.scale(surf, (screen_width, screen_height))
    # 画面に描画
    screen.blit(scaled_surf, (0, 0))
    pygame.display.flip()

# キー入力をアクションに変換する関数
def get_action_from_keys():
    """キーボードの入力からアクションを取得する関数"""
    # キーマッピング: 矢印キーをMsPacmanのアクションに対応させる
    # MsPacmanのアクション: 0:NOOP, 1:UP, 2:RIGHT, 3:LEFT, 4:DOWN
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return 1  # 上
    elif keys[pygame.K_RIGHT]:
        return 2  # 右
    elif keys[pygame.K_LEFT]:
        return 3  # 左
    elif keys[pygame.K_DOWN]:
        return 4  # 下
    else:
        return 0  # アクションなし

# メインループ関数
def main_loop(env, screen):
    """ゲームのメインループ"""
    clock = pygame.time.Clock()
    running = True
    total_reward = 0
    
    # 初期状態の取得と表示
    observation, _ = env.reset()
    update_display(screen, observation)
    
    while running:
        # イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # キー入力からアクションを取得
        action = get_action_from_keys()
        
        # 環境を1ステップ進める
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 画面の更新
        update_display(screen, observation)
        
        # ゲームオーバーの処理
        if terminated or truncated:
            print(f"ゲーム終了！ スコア: {total_reward}")
            observation, _ = env.reset()
            total_reward = 0
        
        # フレームレートの制御
        clock.tick(30)

# メイン関数
def main():
    """メイン関数"""
    try:
        # ゲームの初期化
        env, screen = init_game()
        # メインループの実行
        main_loop(env, screen)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        # 終了処理
        pygame.quit()
        env.close()
        sys.exit()

# プログラムのエントリーポイント
if __name__ == "__main__":
    main()
