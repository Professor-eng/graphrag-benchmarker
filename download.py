import wikipediaapi, os

wiki = wikipediaapi.Wikipedia('HackathonProject/1.0', 'en')

topics = [
    'Cybersecurity', 'Malware', 'Ransomware', 'Phishing',
    'SQL_injection', 'Zero-day_vulnerability', 'Advanced_persistent_threat',
    'Stuxnet', 'WannaCry_ransomware_attack', 'NotPetya',
    'FireEye', 'CrowdStrike', 'Lazarus_Group', 'Botnet',
    'Cyber_attack', 'Computer_virus', 'Denial-of-service_attack',
    'Man-in-the-middle_attack', 'Social_engineering_(security)',
    'Cybercrime', 'Data_breach', 'Identity_theft', 'Spyware',
    'Trojan_horse_(computing)', 'Rootkit', 'Keylogger'
]

os.makedirs('data', exist_ok=True)

for t in topics:
    p = wiki.page(t)
    if p.exists():
        with open(f'data/{t}.txt', 'w', encoding='utf-8') as f:
            f.write(p.text)
        print(f'Downloaded {t}: {len(p.text.split())} words')
    else:
        print(f'NOT FOUND: {t}')