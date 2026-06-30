import pygame
import random
import json
import os
import math

# --- AYARLAR VE SABITLER ---
WIDTH, HEIGHT = 1000, 600
MAP_WIDTH = 25000  # Yaklasik 1 dakika surecek uzun harita
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (150, 150, 150)
BROWN = (139, 69, 19)

MIN_TAKEOFF_SPEED = 3
MAX_LANDING_SPEED = 4
MAX_FUEL = 2500
TIME_LIMIT = 5000  

# --- Q-LEARNING YAPAY ZEKA ---
class QLearningAgent:
    def __init__(self, action_size):
        self.q_table = {}
        self.action_size = action_size
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.02
        self.load_model()

    def get_state_key(self, plane, env):
        dis_y = int(plane.y // 50)
        dis_speed = int(plane.speed)
        dis_angle = int(plane.angle // 10)
        dis_mountain = int((env.mountain_x - plane.x) // 300)
        return f"{dis_y}_{dis_speed}_{plane.gear_deployed}_{dis_angle}_{env.wind_level}_{env.wind_dir}_{plane.direction}_{dis_mountain}"

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_size
        return self.q_table[state].index(max(self.q_table[state]))

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_size
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * self.action_size

        best_next_action = max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        
        self.q_table[state][action] = current_q + self.learning_rate * (reward + self.discount_factor * best_next_action - current_q)

    def save_model(self):
        with open("q_table.json", "w") as f:
            json.dump(self.q_table, f)

    def load_model(self):
        if os.path.exists("q_table.json"):
            with open("q_table.json", "r") as f:
                self.q_table = json.load(f)
            self.epsilon = 0.1

# --- GORSEL OLUSTURUCU ---
def get_plane_image(gear_deployed, crashed):
    surf = pygame.Surface((100, 50), pygame.SRCALPHA)
    
    body_color = (240, 240, 240) if not crashed else (70, 70, 70)
    tail_color = (200, 50, 50) if not crashed else (40, 40, 40)
    wing_color = (150, 150, 150) if not crashed else (50, 50, 50)
    window_color = (50, 50, 50) if not crashed else (20, 20, 20)
    cockpit_color = (100, 200, 255) if not crashed else (40, 60, 80)
    
    # Govde
    pygame.draw.ellipse(surf, body_color, (5, 15, 90, 20))
    # Kokpit Cami
    pygame.draw.polygon(surf, cockpit_color, [(75, 18), (87, 23), (75, 28)])
    # Yolcu Camlari
    for i in range(30, 70, 8):
        pygame.draw.circle(surf, window_color, (i, 23), 2)
    # Kuyruk
    pygame.draw.polygon(surf, tail_color, [(10, 25), (25, 0), (35, 20)])
    # Kanat
    pygame.draw.polygon(surf, wing_color, [(35, 25), (60, 25), (45, 45)])
    
    # Tekerler
    if gear_deployed:
        # Arka teker
        pygame.draw.rect(surf, (100, 100, 100), (25, 30, 4, 12))
        pygame.draw.circle(surf, (20, 20, 20), (27, 44), 6)
        # On teker
        pygame.draw.rect(surf, (100, 100, 100), (75, 30, 4, 12))
        pygame.draw.circle(surf, (20, 20, 20), (77, 44), 6)
        
    return surf

# --- UCAK SINIFI ---
class Plane:
    def __init__(self):
        self.x = 100
        self.y = HEIGHT - 70  # Tekerlerin tam yere degmesi icin ayarlandi
        self.speed = 0
        self.angle = 0  
        self.fuel = MAX_FUEL
        self.gear_deployed = True
        self.direction = 1  
        self.crashed = False
        self.landed = False
        self.crash_reason = ""

    def act(self, action):
        if action == 0 and self.fuel > 0: 
            self.speed = min(20, self.speed + 0.5)
        elif action == 1: 
            self.speed = max(0, self.speed - 0.5)
        elif action == 2: 
            if self.y < HEIGHT - 70 or self.speed >= MIN_TAKEOFF_SPEED:
                self.angle = min(36, self.angle + 1)
        elif action == 3: 
            if self.y < HEIGHT - 70:
                self.angle = max(-36, self.angle - 1)
        elif action == 4: 
            self.gear_deployed = not self.gear_deployed
        elif action == 5: 
            # Hiz 6'dan fazlayken yon degistirilemez
            if self.speed <= 6:
                self.direction *= -1

    def update(self, env):
        if self.crashed or self.landed: return

        # Yakit Tuketimi ve Bitme Durumu
        fuel_consumption = 1
        if self.gear_deployed and self.y < HEIGHT - 100:
            fuel_consumption = 3
        self.fuel = max(0, self.fuel - fuel_consumption)
        
        if self.fuel <= 0:
            self.crashed = True
            self.crash_reason = "Yakit tukendi, ucak dustu!"
            return

        # YENI KURAL: Yogun ruzgari arkasina almis ve havalanmis ucagin hizi 4'ten asagi dusmez
        if self.y < HEIGHT - 70 and env.wind_level == 2 and self.direction == env.wind_dir:
            self.speed = max(4.0, self.speed)

        # Ruzgar Hesaplamasi
        actual_speed = self.speed * self.direction
        if env.wind_level > 0:
            actual_speed += (env.wind_level * 0.5) * env.wind_dir

        # Aciya Gore Ilerleme Fizigi
        rad = math.radians(self.angle)
        self.x += actual_speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)

        # Havada Hiz < 2 Ise Stall (Dusme) Durumu
        if self.y < HEIGHT - 70 and self.speed < 2:
            self.y += 4
            self.angle = max(-35, self.angle - 1)

        # Dag Engeli Kontrolu
        if env.mountain_x <= self.x <= env.mountain_x + env.mountain_width:
            mid = env.mountain_x + env.mountain_width // 2
            if self.x < mid:
                pct = (self.x - env.mountain_x) / (mid - env.mountain_x)
            else:
                pct = (env.mountain_x + env.mountain_width - self.x) / (env.mountain_x + env.mountain_width - mid)
            mountain_y = HEIGHT - 20 - (env.mountain_height * pct)
            if self.y + 25 >= mountain_y:  # Govde temas kontrolu
                self.crashed = True
                self.crash_reason = "Daga carptiniz, ucak patladi!"

        # Yere Temas Kontrolu
        if self.y >= HEIGHT - 70:
            self.y = HEIGHT - 70
            if self.x > 200:  
                self.check_landing(env)
            else:
                self.angle = 0

        # Sinirlar
        self.x = max(0, min(MAP_WIDTH, self.x))
        self.y = max(0, min(HEIGHT - 70, self.y))

    def check_landing(self, env):
        if not self.gear_deployed:
            self.crashed = True
            self.crash_reason = "Tekerler kapali, ucak patladi!"
            return
        
        if abs(self.angle) > 8:
            self.crashed = True
            self.crash_reason = "Ucak yere paralel degil, burnu kirildi!"
            return

        if self.speed > MAX_LANDING_SPEED:
            self.crashed = True
            self.crash_reason = "Cok hizli inis, tekerler yandi!"
            return
        
        if env.wind_level == 2 and self.direction == env.wind_dir:
            self.crashed = True
            self.crash_reason = "Yogun ruzgara ters yonde inis yapmalisin, tekerler yandi!"
            return

        # Basarili Durma / Varis Kontrolu
        if self.x > MAP_WIDTH - 1000:
            self.landed = True
        else:
            self.angle = 0  

# --- CEVRE (ENVIRONMENT) ---
class Environment:
    def __init__(self):
        self.wind_level = random.choice([0, 1, 2])
        self.wind_dir = random.choice([-1, 1])
        self.timer = TIME_LIMIT
        
        self.trees = [random.randint(1500, MAP_WIDTH - 2000) for _ in range(40)]
        self.lakes = [random.randint(2000, MAP_WIDTH - 3000) for _ in range(12)]
        
        # Rastgele bulut koordinatlari (X, Y)
        self.clouds = [(random.randint(0, MAP_WIDTH), random.randint(20, 250)) for _ in range(20)]
        
        self.mountain_x = 11000
        self.mountain_width = 800
        self.mountain_height = 280

def draw_text(surf, text, x, y, color=BLACK, size=24):
    font = pygame.font.SysFont(None, size)
    img = font.render(text, True, color)
    surf.blit(img, (x, y))

# --- ANA DONGU ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Ucak Simulasyonu")
    clock = pygame.time.Clock()

    ai_agent = QLearningAgent(action_size=7)
    
    # --- MENU ARKA PLANINI YUKLEME ---
    try:
        # r"" kullanarak Windows dosya yolundaki \ isaretlerinin sorun cikarmasini engelliyoruz
        menu_bg = pygame.image.load(r"C:\logopilot.bmp")
        # Resmi tam pencere boyutuna (1000x600) gore esnetme
        menu_bg = pygame.transform.scale(menu_bg, (WIDTH, HEIGHT))
    except Exception as e:
        print(f"Uyari: Arka plan resmi yuklenemedi! Hata: {e}")
        menu_bg = None # Hata olursa oyun cokmesin diye yedek plan

    master_running = True
    info_screen = False  # YENI DEGISKEN: Bilgi ekrani acik mi kapali mi?

    while master_running:
        mode = None
        while mode is None and master_running:
            
            # --- BILGI EKRANI (B TUSUNA BASILDIYSA BURASI CIZILIR) ---
            if info_screen:
                screen.fill((40, 40, 40))  # Koyu gri arka plan
                draw_text(screen, "--- UCUS KILAVUZU ---", WIDTH//2 - 100, 40, (255, 215, 0))

                # Kontroller
                draw_text(screen, "[ TUS KONTROLLERI ]", 50, 100, (0, 255, 0))
                draw_text(screen, "Sag / Sol Ok    : Motor gucu arttir/azalt. (ucak yonu degisse de tus yonu degismez, hizlanma koluyla ayni mantik)", 50, 130, WHITE)
                draw_text(screen, "Yukari / Asagi Ok: Kuyrugu arala (Yuksel) / Kanadi arala (Alcal)", 50, 160, WHITE)
                draw_text(screen, "G Tusu          : Inis Takimlarini (Tekerleri) Ac / Kapat", 50, 190, WHITE)
                draw_text(screen, "Bosluk (Space)  : Yon Degistir (Hiz 6(450km/h) ve altindayken yapilabilir)", 50, 220, WHITE)
                draw_text(screen, "ESC Tusu        : Oyundan cikip menuye don", 50, 250, WHITE)

                # Kurallar ve Yasaklar
                draw_text(screen, "[ DIKKAT EDILMESI GEREKENLER ]", 50, 310, (255, 100, 100))
                draw_text(screen, "- Tekerlekler acikken ucarsan hava turbulansi yuzunden 3 kat fazla yakit harcarsin.", 50, 340, WHITE)
                draw_text(screen, "- Havada hizini 2(150km/h)'nin altina dusurursen Ucak tutunamaz ve dusmeye baslar.", 50, 370, WHITE)
                draw_text(screen, "- Iniste ucak yere olabildigince paralel olsun, sanayide ucak tamponu bulmak zor.", 50, 400, WHITE)
                draw_text(screen, "- Hiz 4(300km/h)'ten buyukken yere degersen tekerlekler parcalanir.", 50, 430, WHITE)
                draw_text(screen, "- Cok ruzgar varsa ruzgara karsi donup oyle inmelisiniz.", 50, 460, WHITE)
                draw_text(screen, "- Sondaki yesil yere kazasiz belasiz inerseniz kazanirsiniz", 50, 490, WHITE)
                draw_text(screen, "- Eray Yasin CACA", 50, 520, WHITE)

                draw_text(screen, "Geri donmek icin 'B' veya 'ESC' tusuna basin", WIDTH//2 - 180, HEIGHT - 50, (255, 215, 0))
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        master_running = False
                        mode = "EXIT"
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_b or event.key == pygame.K_ESCAPE:
                            info_screen = False  # Bilgi ekranini kapat
            
            # --- NORMAL MENU (BILGI EKRANI KAPALIYSA BURASI CIZILIR) ---
            else:
                if menu_bg:
                    screen.blit(menu_bg, (0, 0)) 
                else:
                    screen.fill((135, 206, 235)) 
                
                draw_text(screen, "ISTIKBAL GOKLERDE", WIDTH//2 - 90, HEIGHT//5, WHITE)
                draw_text(screen, "Manuel oynamak icin 'M' tusuna basin", WIDTH//2 - 140, HEIGHT//2)
                draw_text(screen, "Yapay Zeka (Ogrenme) icin 'A' tusuna basin", WIDTH//2 - 160, HEIGHT//2 + 30)
                draw_text(screen, "Yapay Zekaya Ogretmek icin 'T' tusuna basin", WIDTH//2 - 165, HEIGHT//2 + 60)
                
                # YENI: Bilgiler menusu yonlendirmesi
                draw_text(screen, "Nasil Oynanir? (Bilgiler) icin 'B' tusuna basin", WIDTH//2 - 180, HEIGHT//2 + 120, (255, 215, 0))
                
                pygame.display.flip()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        master_running = False
                        mode = "EXIT"
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_m: mode = "MANUAL"
                        elif event.key == pygame.K_a: mode = "AI"
                        elif event.key == pygame.K_t: mode = "TEACH"
                        elif event.key == pygame.K_b: info_screen = True  # Bilgi ekranini ac

        if not master_running or mode == "EXIT": break

        plane = Plane()
        env = Environment()
        episode = 1
        total_reward = 0
        game_running = True
        
        # OGRETICI MOD ICIN HAFIZA
        teaching_memory = [] 

        while game_running:
            if mode == "AI": clock.tick(400)  
            else: clock.tick(FPS)
            
            screen.fill((135, 206, 235))
            env.timer -= 1

            state = ai_agent.get_state_key(plane, env)
            action = 6  

            keys = pygame.key.get_pressed()
            if mode in ["MANUAL", "TEACH"]:
                if keys[pygame.K_RIGHT]: action = 0
                elif keys[pygame.K_LEFT]: action = 1
                elif keys[pygame.K_UP]: action = 2
                elif keys[pygame.K_DOWN]: action = 3

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    ai_agent.save_model()
                    game_running = False
                    master_running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:  
                        game_running = False
                    
                    # AI'in bu eylemleri de ogrenebilmesi icin action degiskenine baglandi
                    if mode in ["MANUAL", "TEACH"]:
                        if event.key == pygame.K_g: action = 4  
                        elif event.key == pygame.K_SPACE: action = 5  

            if mode == "AI" and not plane.crashed and not plane.landed:
                action = ai_agent.get_action(state)

            if mode in ["MANUAL", "AI", "TEACH"]:
                plane.act(action)
            
            plane.update(env)

            # --- AI VE OGRETME ODUL MEKANIZMASI ---
            if mode in ["AI", "TEACH"]:
                reward = 0
                if plane.crashed: 
                    progress_bonus = (plane.x / MAP_WIDTH) * 1000
                    reward = -1500 + progress_bonus
                elif env.timer <= 0: 
                    reward = -500
                elif plane.landed: 
                    reward = 3000
                else:
                    if plane.direction == 1 and plane.speed > 0: reward += plane.speed * 0.2
                    if plane.direction == -1 and plane.x < MAP_WIDTH - 2000: reward -= 15 
                    if plane.gear_deployed and plane.y < HEIGHT - 200 and plane.x < MAP_WIDTH - 2000: reward -= 5
                    if plane.x >= MAP_WIDTH - 2000 and plane.gear_deployed and plane.speed <= MAX_LANDING_SPEED + 2: reward += 5 
                    if HEIGHT - 500 < plane.y < HEIGHT - 200 and plane.angle == 0 and plane.direction == 1: reward += 1
                    if plane.y < HEIGHT - 70 and plane.speed < 2: reward -= 10 
                    if 0 < (env.mountain_x - plane.x) < 1000 and plane.y > HEIGHT - 200: reward -= 10

                next_state = ai_agent.get_state_key(plane, env)
                
                # Eger AI oynuyorsa aninda ogren, eger sen ogretiyorsan hafizaya al
                if mode == "AI":
                    ai_agent.update_q_value(state, action, reward, next_state)
                    total_reward += reward
                elif mode == "TEACH":
                    teaching_memory.append((state, action, reward, next_state))

            # --- KAMERA VE CIZIMLER ---
            camera_x = plane.x - WIDTH // 4
            camera_x = max(0, min(MAP_WIDTH - WIDTH, camera_x))

            for cx, cy in env.clouds:
                if camera_x - 100 < cx < camera_x + WIDTH + 100:
                    pygame.draw.circle(screen, (245, 245, 245), (cx - camera_x, cy), 30)
                    pygame.draw.circle(screen, (245, 245, 245), (cx + 30 - camera_x, cy - 15), 40)
                    pygame.draw.circle(screen, (245, 245, 245), (cx + 60 - camera_x, cy), 30)

            pygame.draw.rect(screen, GRAY, (0 - camera_x, HEIGHT - 20, MAP_WIDTH, 20))
            pygame.draw.rect(screen, GREEN, (MAP_WIDTH - 1000 - camera_x, HEIGHT - 20, 1000, 20))

            for tx in env.trees:
                if camera_x - 40 < tx < camera_x + WIDTH + 40:
                    pygame.draw.rect(screen, BROWN, (tx - camera_x, HEIGHT - 40, 10, 20))
                    pygame.draw.circle(screen, GREEN, (tx - camera_x + 5, HEIGHT - 45), 15)

            for lx in env.lakes:
                if camera_x - 150 < lx < camera_x + WIDTH + 150:
                    pygame.draw.ellipse(screen, BLUE, (lx - camera_x, HEIGHT - 22, 120, 10))

            mx, mw, mh = env.mountain_x, env.mountain_width, env.mountain_height
            points = [(mx - camera_x, HEIGHT - 20), (mx + mw//2 - camera_x, HEIGHT - 20 - mh), (mx + mw - camera_x, HEIGHT - 20)]
            pygame.draw.polygon(screen, GRAY, points)

            plane_img = get_plane_image(plane.gear_deployed, plane.crashed)
            if plane.direction == -1: plane_img = pygame.transform.flip(plane_img, True, False)
            rot_angle = plane.angle if plane.direction == 1 else -plane.angle
            rotated_plane = pygame.transform.rotate(plane_img, rot_angle)
            
            rect = rotated_plane.get_rect(center=(plane.x - camera_x + 50, plane.y + 25))
            screen.blit(rotated_plane, rect.topleft)

            draw_text(screen, f"Mod: {mode}  |  ESC ile Menuye Don", 10, 10)
            draw_text(screen, f"Hiz: {plane.speed:.1f} ({75 * plane.speed:.1f} km/sa) |  Aci: {plane.angle}", 10, 35)
            draw_text(screen, f"Yakit: {int(plane.fuel)}  |  Mesafe: {int(plane.x)} / {MAP_WIDTH}", 10, 60)
            draw_text(screen, f"Zaman: {env.timer}", 10, 85)
            
            w_level = "Yok" if env.wind_level == 0 else ("Hafif" if env.wind_level == 1 else "Yogun")
            w_dir = "Sag" if env.wind_dir == 1 else "Sol"
            draw_text(screen, f"Ruzgar: {w_level} ({w_dir})", 10, 110)
            
            if mode == "AI":
                draw_text(screen, f"Episode: {episode}  |  Epsilon: {ai_agent.epsilon:.3f}", 10, 135)
            elif mode == "TEACH":
                draw_text(screen, "AI KAYITTA... (Ogretme Modu)", 10, 135, RED)

            # --- OYUN SONU VE OGRETME ONAY MENUSU ---
            if plane.crashed or plane.landed or env.timer <= 0:
                msg = "BASARILI INIS!" if plane.landed else ("SURE BITTI!" if env.timer <= 0 else plane.crash_reason)
                draw_text(screen, msg, WIDTH//2 - 250, HEIGHT//2, RED if not plane.landed else GREEN, size=51)
                pygame.display.flip()
                pygame.time.delay(1200)
                
                # Eger ogretici moddaysak onay isteme
                if mode == "TEACH":
                    choosing = True
                    while choosing:
                        screen.fill((40, 40, 40))
                        draw_text(screen, "UCUS TAMAMLANDI", WIDTH//2 - 100, HEIGHT//3 - 30, WHITE)
                        draw_text(screen, "1 - YAPAY ZEKAYA OGRET (Q-Table'a Kaydet)", WIDTH//2 - 200, HEIGHT//2, GREEN)
                        draw_text(screen, "2 - IPTAL ET VE MENUYE DON", WIDTH//2 - 150, HEIGHT//2 + 40, RED)
                        pygame.display.flip()
                        
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                pygame.quit()
                                exit()
                            if ev.type == pygame.KEYDOWN:
                                if ev.key == pygame.K_1:
                                    # Hafizadaki tum ucusu AI'a besle
                                    for s, a, r, n_s in teaching_memory:
                                        ai_agent.update_q_value(s, a, r, n_s)
                                    ai_agent.save_model()
                                    choosing = False
                                elif ev.key == pygame.K_2:
                                    # Iptal et, hafizayi temizle
                                    choosing = False
                    
                    # Onay sonrasi ana menuye donmek icin
                    game_running = False 
                    mode = None 

                elif mode == "AI":
                    ai_agent.epsilon = max(ai_agent.epsilon_min, ai_agent.epsilon * ai_agent.epsilon_decay)
                    episode += 1
                    if episode % 10 == 0: ai_agent.save_model()
                    plane = Plane()
                    env = Environment()
                else: # Manual mod
                    plane = Plane()
                    env = Environment()

            pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()