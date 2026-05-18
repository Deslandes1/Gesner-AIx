import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import hashlib
import re
import base64
import csv
import io
import os
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ========== DATA DIRECTORY ==========
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")

# ---------- PERSISTENCE FUNCTIONS ----------
def save_training_data():
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.training_data, f, ensure_ascii=False, indent=2)

def load_training_data():
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_dictionaries():
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.dictionaries, f, ensure_ascii=False, indent=2)

def load_dictionaries():
    if os.path.exists(DICT_FILE):
        with open(DICT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"ht": {}, "fr": {}, "en": {}}

def save_voice_cache():
    serializable = {}
    for key, audio_bytes in VOICE_CACHE.items():
        serializable[key] = base64.b64encode(audio_bytes).decode("utf-8")
    with open(VOICE_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

def load_voice_cache():
    if os.path.exists(VOICE_FILE):
        with open(VOICE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache = {}
        for key, b64 in data.items():
            cache[key] = base64.b64decode(b64)
        return cache
    return {}

# ---------- DEFAULT TRAINING FACTS (UPDATED WITH ALL CHAPTER 4, 5 & WORKRISE POLICIES) ----------
def get_default_training_facts():
    return [
        # ---------- PREVIOUS FACTS (keep all) ----------
        "Ti Malice se yon lojisyèl edikatif ki anseye timoun yo Kreyòl Ayisyen atravè jwèt ak istwa.",
        "Ti Malice gen yon liv ki rele 'Ti Malice aprann Kreyòl' ki gen 12 chapit.",
        "Chapit 1 Ti Malice: Alfabè kreyòl la ak pwononsyasyon.",
        "Chapit 2 Ti Malice: Nonm 1 rive 100 an Kreyòl.",
        "Chapit 3 Ti Malice: Koulè ak fòm an Kreyòl.",
        "Chapit 4 Ti Malice: Fanmi ak zanmi.",
        "Chapit 5 Ti Malice: Manje Ayisyen.",
        "Chapit 6 Ti Malice: Bèt ak natir.",
        "Chapit 7 Ti Malice: Vèb ki pi komen yo.",
        "Chapit 8 Ti Malice: Tan pase, tan prezan, tan kap vini.",
        "Chapit 9 Ti Malice: Fraz senp.",
        "Chapit 10 Ti Malice: Konvèsasyon chak jou.",
        "Chapit 11 Ti Malice: Pwovèb ak ekspresyon Kreyòl.",
        "Chapit 12 Ti Malice: Istwa kout pou li.",
        "Ti Malice gen yon seksyon egzèsis ki gen 50 kesyon pou pratike.",
        "Ou ka telechaje Ti Malice sou sitwèb globalinternet.py.",
        "Ti Malice fèt pa Gesner Deslandes pou ede timoun Ayisyen aprann Kreyòl fasilman.",
        "Alfabè kreyòl la gen 32 lèt.",
        "Pwonon pèsonèl an Kreyòl: Mwen, ou, li, nou, yo.",
        "Vèb 'se' (to be) nan prezan: Mwen se, ou se, li se, nou se, yo se.",
        "Vèb 'gen' (to have) nan prezan: Mwen gen, ou gen, li gen, nou gen, yo gen.",
        "Salitasyon debaz: Bonjou, Bonswa, Kijan ou rele?, Mwen rele...",
        "Kesyon debaz: Kijan ou ye?, Mwen byen, Mèsi, Pa dekwa.",
        "Nonm 1-10: youn, de, twa, kat, senk, sis, sèt, uit, nèf, dis.",
        "Koulè debaz: wouj, ble, vèt, jòn, nwa, blan.",
        "Tan pase: yo itilize 'te' devan vèb. Egzanp: Mwen te manje.",
        "Tan kap vini: yo itilize 'ap' oswa 'pral'. Egzanp: Mwen ap manje.",
        "Nègasyon: yo itilize 'pa' apre vèb. Egzanp: Mwen pa manje.",
        "Pwopozisyon: nan, sou, anba, devan, dèyè, bò.",
        "Fraz konplèks: itilize 'ki', 'kote', 'poukisa'.",
        "Vèb modèl: vle, kapab, dwe, konnen, fè.",
        "Pawòl konpoze: pote + chay = potechay, bwa + chemen = bwachemen.",
        "Pwovèb: 'Dèyè mòn gen mòn', 'Men anpil, chay pa lou', 'Ti ponyen fè gwo chay'.",
        "Anplwaye tan ki konpoze: Mwen te ap manje.",
        "Vwa pasif: Liv la te ekri pa Jan.",
        "Sijonktif: Fòk ou vini.",
        "Liteati kreyòl: ekriven tankou Frankétienne, Gary Victor, Lyonel Trouillot.",
        "Diferans ant Kreyòl Ayisyen ak lòt kreyòl.",
        "Analiz powèm: 'Kreyon mwen' pa Gesner Deslandes.",
        "Rédaksyon avançée: kijan pou ekri yon lèt fòmèl an Kreyòl.",
        
        # ---------- WORKRISE POLICIES (existing) ----------
        "Workrise se yon konpayi ki sipòte moun ki fè travay di. Yo byen kontan ou rantre nan ekip la.",
        "Nan Workrise, nou gen yon kilti kote tout moun kolabore, aprann, ak ede youn lòt.",
        "Valè Workrise yo se: Posede Misyon an, Solisyon anvan Ego, Monte Bawo a, Aprann ak Grandi.",
        "Posede Misyon an vle di nou toujou mete misyon an premye, nou pran inisyativ pou rezoud pwoblèm, epi nou posede rezilta yo.",
        "Solisyon anvan Ego vle di nou pran desizyon ki baze sou reyalite ak done, nou respekte youn lòt, epi nou kolabore pou jwenn pi bon solisyon an.",
        "Monte Bawo a vle di nou vize wo, nou toujou ap chache amelyore, epi nou livre solisyon ki ekselan.",
        "Aprann ak Grandi vle di nou aprann nan echèk nou yo, nou kapitalize sou viktwa, nou louvri lespri, rezistan, epi nou chanje rapidman.",
        "Anviwònman travay Workrise se yon espas ki an sekirite e ki pwodiktif. Nou espere ou deplase vit avèk nou.",
        "Nan Workrise, kominikasyon se kle. Sèvi ak tout teknoloji yo bay pou kolabore.",
        "Ou reprezante Workrise sou tout platfòm kominikasyon ou itilize. Tout kominikasyon ou yo kapab gade nenpòt moman.",
        "Liv Referans sa a aplike pou tout anplwaye Workrise atravè tout filiales li yo.",
        "Liv Referans sa a se pa yon kontra travay. Workrise kapab chanje l nenpòt lè san avètisman.",
        "Nan Workrise, travay la se 'at‑will'. Sa vle di ou kapab kite travay la nenpòt lè, epi konpayi an kapab revoke ou nenpòt lè, avèk oswa san avètisman.",
        "Workrise pa fè diskriminasyon sou baz ras, koulè, relijyon, laj, orijin, andikap, gwosès, idantite sèks, seksyalite, oswa estati militè.",
        "Si ou kwè w ap sibi diskriminasyon, ou dwe rapòte sa bay sipèvizè ou oswa Field HR. Tout plent yo pral envestige.",
        "Workrise entèdi tout fòm asèlman (harassment) ki baze sou seks, ras, laj, relijyon, andikap, oswa lòt karakteristik pwoteje.",
        "Asèlman seksyèl gen ladan demand favè seksyèl, komèks ofansif sou kò yon moun, oswa kreye yon anviwònman ostil.",
        "Si w ap temwen asèlman, ou dwe rapòte l imedyatman bay Field HR oswa lè l sèvi avèk liy dirèk Red Flag (1-877-647-3335).",
        "Workrise pran tout mezi pou pwoteje moun ki rapòte asèlman oswa diskriminasyon kont revanj.",
        "Konpayi an fè aranjman rezonab pou anplwaye andikape ki kapab fè travay la. Kontakte Field HR pou demann.",
        "Anplwaye yo gen dwa patisipe nan aktivite pwoteje tankou pote plent bay EEOC, SEC, oswa NLRB. Konpayi an pa ka revanj kont yo.",
        "Workrise gen yon politik 'pòt louvri'. Ou ka pale ak sipèvizè ou, manadjè, oswa Field HR sou nenpòt enkyetid.",
        "Whistleblower (moun ki denonse fwod) yo pwoteje kont revanj. Yo ka rapòte ilegalite nan 1-877-647-3335.",
        "Lafèm ak zam yo entèdi sou lokal Workrise. Tout menas oswa vyolans rapòte imedyatman.",
        "Anplwaye regilye aplentan travay 40 èdtan semèn. Yo kalifye pou tout benefis.",
        "Anplwaye tanporè travay mwens pase 3 mwa. Yo gen PTO men yo pa gen lòt benefis.",
        "Anplwaye ki egzante (exempt) pa gen dwa pou lè siplemantè. Anplwaye ki non‑egzante (nonexempt) gen dwa pou lè siplemantè.",
        "Workrise fè verification background sou tout moun ki aksepte yon òf travay. Si rapò a pa bon, òf la kapab anile.",
        "Nou dwe verifye dwa ou pou travay Ozetazini nan twa jou apre premye jou travay ou.",
        "Anplwaye yo dwe enfòme direktè yo de chanjman adrès, nimewo, oswa sitiyasyon familyal.",
        "Referans pou ansyen anplwaye: Workrise sèlman bay dat travay ak tit pòsyon. Salè bay sèlman si ansyen anplwaye bay otorizasyon ekri.",
        "Yon anplwaye pa ka sipèvize yon manm fanmi. Si sa rive, youn nan yo kapab transfere oswa revoke.",
        "Pandan w ap travay, ou pa dwe fè komès ak enfòmasyon konfidansyèl sou Workrise. Sa vyole lwa sou komès anndan.",
        "Anplwaye non‑egzante dwe anrejistre tout èdtan travay yo. Yo gen dwa pou repo manje ak repo pandan jounen an.",
        "Anplwaye non‑egzante ki travay plis pase 40 èdtan nan yon semèn gen dwa pou lè siplemantè a 1.5 fwa salè regilye yo.",
        "Anplwaye egzante pa gen dwa pou lè siplemantè men yo toujou gen pou yo fini travay yo.",
        "Salè yo peye chak semèn. Ou ka wè dediksyon sou fèy salè ou.",
        "Konduit entèdi: vòl, fo, fraz menasan, move itilizasyon pwopriyete konpayi, oswa enfliyans sou lòt anplwaye.",
        "Pandan w pa nan travay, ou dwe evite konduit ki kapab afekte non Workrise oswa anpeche lòt anplwaye travay.",
        "Fimen ak itilize dwòg ilegal entèdi sou lokal Workrise. Alkòl pèmèt sèlman nan evènman otorize.",
        "Anplwaye yo dwe konfòme yo ak orè yo. Absans san avètisman ka mennen nan revokasyon.",
        "Nan ka ijans (tranblemanntè, move tan), Workrise kapab fèmen. Anplwaye dwe kontakte sipèvizè yo.",
        "Enfòmasyon konfidansyèl (done kliyan, estrateji) pa dwe pataje san otorizasyon.",
        "Telefòn ak aparèy pèsonèl yo dwe itilize avèk jijman. Anplwaye yo pa dwe gen apèl twò long oswa voye SMS pandan lè travay.",
        "Depans biznis (vwayaj, repa) kapab ranbouse si yo apwouve davans. Mande sipèvizè ou.",
        "Pwopriyete konpayi (laptop, telefòn) yo dwe itilize sèlman pou travay. Pa enstale lojisyèl san pèmisyon.",
        "Anplwaye yo gen aksè sèlman nan zòn yo bezwen. Kat aksè pa dwe prete.",
        "Yo pa gen dwa itilize lokal Workrise lè yo pa nan travay san otorizasyon.",
        "Anplwaye dwe rapòte nenpòt aksidan oswa blesi imedyatman. Sekirite se responsablite tout moun.",
        "Yo pa gen dwa solisite oswa distribye literati politik sou lokal travay san pèmisyon.",
        "Workrise gen kamera sekirite pou pwoteje moun ak byen. Anplwaye yo dwe konnen sa.",
        "Anplwaye aplentan (40 èdtan) gen dwa pou asirans sante, PTO, ak lòt benefis.",
        "Anplwaye a tan pasyèl (30-40 èdtan) gen dwa pou PTO ak asirans sante, men pwopòsyonèl.",
        "PTO (Paid Time Off) – kantite jou depann sou ansyènte. Ou dwe fè demann davans.",
        "Lwa FMLA bay moun ki malad, ki fèk fè yon tibebe, oswa k ap pran swen yon fanmi – 12 semèn konje san salè. Yo dwe travay 12 mwa ak 1250 èdtan pou kalifye.",
        "Fanm ansent gen dwa pran repo pou bay tete. Workrise dwe bay yon espas prive.",
        "Anplwaye ki nan rezèv militè yo gen dwa pou konje militè san pèdi travay yo.",
        "Anplwaye gen dwa pou konje pou ale vote. Fè demann nan Field HR.",
        "Konje pou jiri (jury duty) pèmèt. Workrise peye diferans lan si jiri la pa peye.",
        "Viktim krim kapab pran konje pou ale nan tribinal. Kontakte Field HR.",
        "Workrise gen konpansasyon aksidan travay. Si ou blese nan travay, rapòte imedyatman.",
        "Nenpòt diskisyon ant anplwaye ak Workrise dwe ale nan abitraj (arbitration), pa nan tribinal. Sa vle di ou renonse dwa pou jiri.",
        "Ou gen dwa pale ak yon avoka anvan ou siyen akò abitraj la.",
        "Men ou ka toujou pote plent bay EEOC, NLRB, oswa lòt ajans gouvènman yo.",
        "Workrise gen yon sant èd pou anplwaye (Field HR) ki disponib pou reponn kesyon. Imèl yo se fieldhr@workrise.com.",
        "Ou ka fè rapò anonim atravè Red Flag Reporting: 1-877-647-3335 oswa redflag@redflagreporting.com.",
        "Anplwaye dwe li tout Liv Referans sa a. Si ou gen kesyon, mande Field HR.",
        "Si yon anplwaye Workrise kite travay li volontèman, li dwe remèt tout pwopriyete konpayi an imedyatman.",
        "Si yon anplwaye pa vini travay pandan 3 jou san li pa avèti sipèvizè li, sa konsidere kòm demisyon volontè.",
        "Tout bagay ki pou konpayi an (laptop, badj, kle, inifòm, katy kredi, ekipman sekirite) dwe retounen lè w sispann travay.",
        "Vyolasyon règleman Workrise kapab mennen nan sanksyon jiska revokasyon.",
        "Anplwaye travayè yo ap separe administrativman 30 jou apre dènye jou travay yo, si yo pa resevwa nouvo pwojè oswa yo pa nan konje apwouve.",
        "Si w ap kite Workrise, mande ou fè sa pa yon lèt ekri ak rezon, dat, ak nouvo adrès.",
        "Workrise apresye ke ou bay avètisman pi vit posib pou yo kapab ranplase w.",
        "Si w kite travay san avètisman, w ap resevwa dènye chèk ou nan biwo a. Ou pap kapab rekite ankò si ou pa bay avètisman.",
        "Lè yon pwojè fini, Workrise ap chèche lòt pwojè pou kenbe w travay.",
        "Si w deside kite yon pwojè anvan l fini, w pap ranbouse pou demobilizasyon. Ou dwe retounen tout ekipman an.",
        "Orè travay la fikse pa kliyan an pou chak pwojè. Semèn travay la kòmanse dimanch 12:00 AM CST epi li fini samdi 11:59 PM CST.",
        "Anplwaye travayè yo peye chak semèn oswa chak de semèn, dapre lèt òf la. Si jou peman tonbe nan wikenn oswa jou ferye, yo peye anvan.",
        "Anplwaye ki non‑egzante dwe anrejistre tout èdtan travay yo chak jou. Yo dwe chache apwobasyon sipèvizè anvan yo fè lè siplemantè.",
        "Menm si yo pa resevwa apwobasyon davans, yo dwe rapòte tout èdtan yo travay epi yo p ap peye lè siplemantè san apwobasyon.",
        "Ou dwe anrejistre lè w kòmanse, lè w kanpe pou repo, lè w kòmanse ankò, ak lè w sispann nan fen jounen an.",
        "Repo kout yo peye si yo mwens pase 20 minit. Pa janm anrejistre tan pou yon lòt anplwaye. Fòsifikasyon tan se yon vyolasyon grav.",
        "Si w wè yon erè nan tan ou, kontakte sipèvizè ou imedyatman pou l kominike ak Payroll.",
        "Anplwaye non‑egzante gen dwa pou de repo peye 10 minit chak jou. Yo gen dwa pou 30 minit repo pou manje ki pa peye.",
        "Anplwaye non‑egzante peye pou lè siplemantè a 1.5 fwa salè regilye. Yo dwe gen apwobasyon ekri sipèvizè anvan yo fè lè siplemantè.",
        "Lè siplemantè kòmanse apre 40 èdtan travay nan yon semèn. Jou konje (PTO, vakans, maladi) pa konte nan 40 èdtan sa yo.",
        "Anplwaye egzante pa gen lè siplemantè; salè yo kouvri lè siplemantè si bezwen.",
        "Dediksyon obligatwa yo: taks federal ak eta, sekirite sosyal, Medicare, ak lòt dediksyon lalwa mande.",
        "Lòt dediksyon (asirans, charite, ekipman) dwe gen otorizasyon ekri ou. Ou ka kontakte Payroll pou chanje alokasyon taks ou.",
        "Si w pa retounen pwopriyete konpayi an, Workrise kapab pran sou chèk ou pou sa.",
        "Anplwaye egzante (salary) ap resevwa salè yo pou nenpòt semèn kote yo fè yon ti travay. Semèn kòmanse dimanch 12:00 AM rive samdi 11:59 PM.",
        "Workrise konfòme ak lwa federal (FLSA) ak lwa eta konsènan dediksyon salè pou anplwaye egzante.",
        "Dediksyon ki pa otorize pa lwa FLSA entèdi. Si ou kwè gen yon dediksyon ki pa bon, rapòte sa bay Field HR imedyatman.",
        "Field HR ap envestige tout rapò sou dediksyon ki pa bon. Si yo jwenn erè, w ap ranbouse epi yo pran mezi pou evite sa rive ankò.",
        "Anplwaye Workrise dwe diskite ak sipèvizè yo anvan yo kòmanse yon lòt travay deyò. Sa gen ladan sèvi nan komisyon piblik oswa gouvènman.",
        "Workrise bezwen konnen tout lòt travay anplwaye yo fè pou evite konfli enterè.",
        "Si ou pa jwenn apwobasyon avan ou kòmanse yon lòt travay, ou kapab sibi sanksyon jiska revokasyon.",
        
        # ========== CHAPTER 4 – STANDARDS OF CONDUCT ==========
        "Workrise entèdi fo dokiman, vòl, domaj pwopriyete, itilizasyon san otorizasyon, vyolasyon sekirite, batay, pote zam, konduit kriminèl.",
        "Si ou refize obeyi sipèvizè ou, ou kapab sibi sanksyon.",
        "Ou dwe avèti sipèvizè ou anvan ou absan. Absans san rezon legal ka mennen nan revokasyon.",
        "Ou pa gen dwa kite travay san pèmisyon pandan lè travay (sèlman repo ak manje).",
        "Fè lè siplemantè san apwobasyon entèdi. Refize fè lè siplemantè asiyen tou entèdi.",
        "Workrise pa chanje politik at‑will. Ou toujou kapab kite travay nenpòt lè.",
        "Konduit ilegal pandan w pa nan travay ki afekte Workrise oswa kapasite w pou fè travay w ap tolere.",
        "Dezyèm travay dekouraje anpil. Entèdi si l kreye konfli, afekte pèfòmans, oswa konpetisyon ak Workrise.",
        "Fimen (vap, sigarèt elektwonik) entèdi andedan lokal Workrise oswa nan 20 pye nan pòt, fenèt, oswa priz lè.",
        "Pa travay anba enfliyans dwòg ilegal, alkòl, oswa medikaman ki afekte kapasite w.",
        "Itilizasyon marijana rekreyatif oswa medikal entèdi menm nan eta legal.",
        "Workrise ka fè tès dwòg ak alkòl anvan travay, apre aksidan, oswa oaza. Refi tès se rezon pou revokasyon.",
        "Si ou kondane pou vyolasyon lwa sou dwòg oswa alkòl, ou dwe notifye Field HR nan 5 jou, sinon ou kapab revoke.",
        "Asiduite ak ponktyalite esansyèl. Si ou pa kapab vini, avèti sipèvizè ou omwen yon èdtan anvan.",
        "Move tandans prezans (reta, kite bonè, absans repete) kapab mennen nan revokasyon.",
        "Si move tan anpeche w vini, avèti sipèvizè ou. Ou dwe vini le kondisyon amelyore.",
        "Si Workrise fèmen pou lajounen, yo ap eseye notifye ou.",
        "Ou dwe siyen yon akò konfidansyalite anvan ou kòmanse travay. Pa divilge enfòmasyon konfidansyèl san otorizasyon.",
        "Obligasyon konfidansyalite kontinye apre w fin kite Workrise.",
        "Workrise entèdi koripsyon, trayizon, konfli enterè. Pa bay kado oswa peman san apwobasyon.",
        "Pa aksepte kado nan men moun k ap postile pou travay. Sa kapab revokasyon.",
        "Pandan w ap condui, itilize telefòn men lib (hands‑free). Pa voye SMS. Pa reponn apèl biznis pandan w ap condui.",
        "Vyolasyon sa kapab revokasyon. Workrise pap peye amann pou itilizasyon telefòn.",
        "Lè w itilize machin pèsonèl pou travay, ou dwe gen lisans ak asirans. Workrise peye yon alokasyon kilométraj (IRS rate).",
        
        # ========== CHAPTER 5 – OPERATIONAL CONSIDERATIONS ==========
        "Workrise ranbouse depans biznis rezonab. Soumèt depans ak resi nan 30 jou. Ranbousman fèt nan Payroll.",
        "Anplwaye ki vwayaje resevwa per diem pou manje, lojman, ak depans pandan vwayaj. Montan varye selon pwojè.",
        "Pwopriyete Workrise (laptop, biwo, sistèm) dwe itilize sèlman pou travay. Pa espere vi prive. Workrise kapab enspekte nenpòt lè.",
        "Itilizasyon entènèt ak sistèm Workrise dwe pou travay. Pa enstale lojisyèl san otorizasyon.",
        "Pa voyè spam, èrèl mas, oswa atake sistèm. Pa pataje modpas ou.",
        "Pa pibliye enfòmasyon konfidansyèl sou rezo sosyal. Deklare ke opinyon pa fòtman reprezante Workrise.",
        "Workrise ka kontwole tout itilizasyon sistèm. Vyolasyon kapab revokasyon.",
        "Pa pote zam sou lokal. Pa pataje modpas. Pa espere vi prive nan emèl oswa ot sistèm Workrise.",
        "Pa itilize lokal Workrise oswa pwopriyete konpayi lè w pa nan travay.",
        "Workrise ka enspekte bagaj pèsonèl si gen dout rezonab sou vòl oswa dwòg.",
        "Rapòte moun ki san rezon. Sekirize biwo w lè w ale.",
        "Vizitè dwe siyen ak pase badj. Anplwaye dwe akonpaye yo tout tan.",
        "Rapòte tout kondisyon danjere. Pa janm revanj kont moun ki rapòte. Rapòte blesi imedyatman.",
        "Pa solisite oswa distribye literati pandan lè travay. Entèdi pou non‑anplwaye.",
        "Workrise ka enstale kamera sekirite nan zòn travay. Pa espere vi prive. Pa gen kamera nan twalèt oswa chanje rad.",
        
        # ========== NEW WORKRISE POLICY FACTS (from 5.13, 6.01-6.12, Arbitration) ==========
        "Workrise pap peye konpansasyon aksidan travay pou blesi ki rive pandan anplwaye patisipe volontè nan aktivite lwazi, sosyal, oswa espòtif lè li pa nan travay, si aktivite sa a pa fè pati devwa travay li.",
        "Workrise bay benefis pou anplwaye regilye aplentan (full-time) ak anplwaye a tan pasyèl (part-time) k ap travay 30 èdtan oswa plis pa semèn. Anplwaye tanporè pa kalifye pou benefis eksepte PTO.",
        "Definisyon 'Domestic Partner' nan Workrise: yon moun menm sèks oswa diferan sèks ak anplwaye a, ki gen 18 an oswa plis, ki pa marye, k ap viv nan menm kay, epi ki pa gen relasyon fanmi ki ta anpeche maryaj nan eta kote yo rete a.",
        "Workrise bay konje (leaves of absence) selon lwa ak politik konpayi pou anplwaye ki andikape, ki bezwen konje long pou devwa sivik, swen fanmi, oswa rezon pèsonèl. Ou dwe notifye manadjè oswa Field HR pi vit posib.",
        "Si ou pa retounen nan travay apre konje a fini, sa pral konsidere kòm yon demisyon volontè. Lè w retounen apre yon konje medikal, ou dwe bay yon sètifikasyon doktè ki di ou kapab travay.",
        "Kantite PTO pou anplwaye travayè Workrise ekri nan lèt òf la. PTO a gen ladann nenpòt konje maladi peye dapre lwa eta a. Anplwaye travayè ka roule (roll over) jiska 80 èdtan PTO ki pa itilize nan fen chak ane.",
        "Ou ka itilize PTO pou nenpòt rezon: repo, detant, maladi, sante mantal, aktivite lekòl timoun, oswa vakans. Ou pa gen dwa gen balans PTO negatif.",
        "Pou mande PTO, fè demann nan Paylocity oswa sistèm HRIS la. Si ou pran plis pase 3 jou pou rekipere apre maladi, Workrise ka mande dokimantasyon.",
        "PTO ki pa itilize yo pa peye lè w separe ak konpayi an, sof si lwa eta a egzije. Men si w retravay nan konpayi an nan 1 an, w ap jwenn balans PTO ou te genyen an.",
        "Dapre FMLA, Workrise bay jiska 12 semèn konje nan yon peryòd 12 mwa pou anplwaye ki kalifye, oswa jiska 26 semèn pou konje pou pran swen yon manm militè blese.",
        "Pou kalifye pou FMLA, ou dwe travay pou Workrise omwen 12 mwa, travay 1250 èdtan nan 12 mwa anvan konje a, epi travay nan yon lokal ki gen omwen 50 anplwaye nan 75 mil.",
        "Rezon pou FMLA: nesans yon timoun, adopsyon oswa plasman nan fanmi, pran swen yon konjwen, timoun oswa paran ki gen yon pwoblèm sante grav, ou menm ki gen yon pwoblèm sante grav, oswa ijans militè.",
        "Workrise mezire 12 mwa FMLA kòm yon peryòd glisman (rolling backward). Ou ka pran konje FMLA kontinyèlman oswa pa ti bout (intermittent).",
        "Pandan w nan konje FMLA, Workrise kontinye benefis sante ou nan menm nivo. Si w pa peye pati ou nan prim asirans, konpayi an ka sispann kouvèti apre 30 jou.",
        "Si w pa retounen travay apre FMLA pou rezon ki pa sante ou oswa fanmi w, Workrise ka mande w ranbouse prim asirans sante konpayi an te peye pandan konje a.",
        "Anplwaye ki pran FMLA dwe bay sipèvizè oswa Field HR yon avi vèbal oswa ekri. Si konje a previzib, bay 30 jou avans.",
        "Workrise bay repo rezonab pou fanm k ap bay tete pou yo ka eksprime lèt. Gen yon chanm prive ak ti frijidè pou estoke lèt. Ou dwe make lèt la ak non ou ak dat. Repo plis pase 20 minit pa peye pou anplwaye non-egzante.",
        "Workrise bay konje militè pou manm Rezèv oswa Gad Nasyonal ki patisipe nan antrennman oswa devwa aktif. Konje a dapre lwa federal ak eta. Workrise pa fè diskriminasyon kont moun ki nan sèvis militè.",
        "Workrise bay konje peye oswa san peye pou ale vote si biwo vòt yo pa louvri omwen 3 anvan oswa apre chèf travay ou. Fè demann nan Paylocity omwen 2 jou davans.",
        "Workrise bay konje peye oswa san peye pou sèvi nan jiri selon lwa eta a. Prezante konvokasyon jiri a pi vit posib. Ou ka itilize PTO pandan konje jiri.",
        "Workrise bay konje san peye pou ponpye volontè, ofisye rezèv lapè, oswa sekouris ki reponn a ijans oswa ki patisipe nan fòmasyon planifye (jiska 14 jou pa ane). Anplwaye travayè ka itilize PTO pandan konje sa a.",
        "Workrise bay konje san peye pou viktim krim oswa fanmi pwòch ki dwe ale nan tribinal. Ou ka itilize PTO pandan konje a. Bay dokimantasyon (konvokasyon, kòmandman tribinal).",
        "Gen lòt kalite konje dapre lwa eta. Gade apendis eta yo nan Liv Referans. Kontakte Field HR pou kesyon.",
        "Si ou blese nan travay, Workrise gen asirans konpansasyon travayè. Rapòte blesi a imedyatman bay manadjè ou oswa hsereporting@workrise.com. Rele Axiom nan (281) 419-7063 pou yon infimyè avant ou ale doktè.",
        "Konpansasyon travayè ka bay swen medikal, lajan kach san taks pou ranplase salè pèdi, ak reyabilitasyon pwofesyonèl.",
        "Lè w retounen apre yon blesi travay, Workrise ofri w menm pòsyon oswa yon pòsyon ekivalan, sof si sa ta afekte operasyon sekirite ak efikasite konpayi an.",
        "Workrise gen yon Akò Abitraj Mityèl. Sa vle di ou ak konpayi an dakò pou rezoud tout diskisyon ki gen rapò ak travay atravè abitraj, pa nan tribinal devan yon jiri. Lwa Federal sou Abitraj (FAA) aplike.",
        "Akò abitraj la kouvri tout reklamasyon sot pase ak lavni: diskriminasyon, asèlman, revanj, salè, lè siplemantè, benefis, ak lòt lit anplwaye. Men ou ka toujou pote plent bay EEOC, NLRB, oswa lòt ajans.",
        "Pou kòmanse abitraj, ou dwei premye swiv Pwosedi Pre-Abitraj: voye yon lèt bay Depatman Legal Workrise, Lè sa a, medyasyon. Si sa pa mache, ou ka depoze yon demand abitraj nan AAA (American Arbitration Association).",
        "Detèminasyon abitraj la dwe fèt nan Austin, Texas, oswa nan zòn metwopolitèn ki pi pre kote ou te travay. Chak pati peye pwòp avoka yo, men Workrise peye pifò frè abitraj eksepte $200 frè depo si anplwaye a kòmanse abitraj la.",
        "Nan abitraj, ou pa ka fè yon aksyon klas (class action) oswa kolektif. Ou dwe pote reklamasyon ou sèlman nan non pèsonèl ou. Anyen nan akò sa a pa anpeche w pale ak yon avoka.",
        "Liv Referans Workrise a pa yon kontra travay. Travay la se 'at-will', sa vle di ou kapab kite travay nenpòt lè, epi Workrise kapab revoke ou nenpòt lè, avèk oswa san rezon.",
        "Lè w siyen Akonisman Liv Referans la, ou konfime ou te resevwa ak li Liv la, ou konprann règleman yo ka chanje, epi ou dakò ak Akò Abitraj Mityèl la. Si w kontinye travay plis pase 45 jou apre ou resevwa akò a, ou konsidere kòm ou aksepte l.",
    ]

def initialize_default_training():
    if not st.session_state.training_data:
        default_facts = get_default_training_facts()
        for fact in default_facts:
            if fact.strip():
                embedding = st.session_state.embedding_model.encode([fact])[0]
                st.session_state.training_data.append({"text": fact, "embedding": embedding.tolist()})
        rebuild_index()
        save_training_data()

# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

# ---------- CSS (same as before) ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
        border-right: 2px solid #e94560;
    }
    .stMarkdown, .stTextInput label, .stTextArea label, .stSelectbox label, .stButton button, .stCaption,
    h1, h2, h3, h4, h5, h6, p, li, div, span, strong, em, .footer,
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSelectbox {
        background-color: #000000 !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #000000 !important;
        border: 1px solid #e94560 !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #000000 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox svg {
        fill: #e94560 !important;
        stroke: #e94560 !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #000000 !important;
        border: 1px solid #e94560 !important;
    }
    div[data-baseweb="popover"] li {
        color: white !important;
        background-color: #000000 !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #e94560 !important;
        color: white !important;
    }
    .stButton button {
        background-color: #e94560 !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        border: none;
    }
    .chat-row .stButton button {
        background-color: #ffaa33 !important;
        padding: 0px 8px !important;
        border-radius: 20px !important;
        font-size: 1rem !important;
        width: auto !important;
        min-width: 40px;
    }
    .chat-row .stButton button:hover {
        background-color: #ffcc66 !important;
        transform: scale(1.02);
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460 !important;
        color: white !important;
        border-radius: 12px;
        border: 1px solid #e94560;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .user-message {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        color: white;
    }
    .assistant-message {
        background: linear-gradient(135deg, #0f3460, #1a4a7a);
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e94560;
    }
    .char-picker {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .char-btn {
        background-color: #2a5298;
        border: none;
        border-radius: 20px;
        padding: 5px 12px;
        color: white;
        cursor: pointer;
        font-size: 1rem;
        transition: 0.2s;
        margin-right: 5px;
    }
    .char-btn:hover {
        background-color: #e94560;
    }
    @keyframes spin-globe {
        0% { transform: rotate(0deg); filter: drop-shadow(0 0 2px gold); }
        50% { filter: drop-shadow(0 0 15px #ffaa33) drop-shadow(0 0 5px orange); }
        100% { transform: rotate(360deg); filter: drop-shadow(0 0 2px gold); }
    }
    .spinning-brain {
        animation: spin-globe 3s linear infinite;
        display: inline-block;
        font-size: 3rem;
        text-align: center;
        width: 100%;
    }
    .sidebar-info {
        text-align: center;
        margin-top: 1rem;
        padding: 0.5rem;
        border-top: 1px solid #e94560;
        font-size: 0.9rem;
    }
    .sidebar-info a {
        color: #ffaa33 !important;
        text-decoration: none;
    }
    .sidebar-info a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- LANGUAGES AND TEXTS ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht",
    "Español": "es"
}

TEXTS = {
    "en": {
        "app_title": "Gesner AI - Kreyòl Assistant",
        "chat_input": "Ask me anything in Kreyòl...",
        "send": "Send",
        "clear": "Clear Chat",
        "dictionary": "Dictionary",
        "voice_training": "Voice Training",
        "bulk_training": "Bulk Training",
        "manage_facts": "Manage Facts",
        "test_training": "Test Training",
        "training_center": "Training Center",
        "train_new": "Train New Fact",
        "fact_text": "Fact text",
        "add_fact": "Add Fact",
        "upload_csv": "Upload CSV",
        "upload_audio": "Upload Audio",
        "record_voice": "Record Voice",
        "save_voice": "Save Voice",
        "play": "Play",
        "delete": "Delete",
        "edit": "Edit",
        "update": "Update",
        "chat_interface_label": "Chat"
    },
    "fr": {
        "app_title": "Gesner IA - Assistant Kreyòl",
        "chat_input": "Posez-moi une question en Kreyòl...",
        "send": "Envoyer",
        "clear": "Effacer",
        "dictionary": "Dictionnaire",
        "voice_training": "Entraînement vocal",
        "bulk_training": "Formation en masse",
        "manage_facts": "Gérer les faits",
        "test_training": "Tester l'entraînement",
        "training_center": "Centre de formation",
        "train_new": "Ajouter un fait",
        "fact_text": "Texte du fait",
        "add_fact": "Ajouter",
        "upload_csv": "Importer CSV",
        "upload_audio": "Importer audio",
        "record_voice": "Enregistrer",
        "save_voice": "Sauvegarder",
        "play": "Écouter",
        "delete": "Supprimer",
        "edit": "Modifier",
        "update": "Mettre à jour",
        "chat_interface_label": "Discussion"
    },
    "ht": {
        "app_title": "Gesner AI - Asistan Kreyòl",
        "chat_input": "Pose m yon kesyon an Kreyòl...",
        "send": "Voye",
        "clear": "Efase",
        "dictionary": "Diksyonè",
        "voice_training": "Fòmasyon Vwa",
        "bulk_training": "Fòmasyon an mas",
        "manage_facts": "Jere reyalite yo",
        "test_training": "Tès fòmasyon",
        "training_center": "Sant Fòmasyon",
        "train_new": "Anseye yon nouvo reyalite",
        "fact_text": "Tèks reyalite a",
        "add_fact": "Ajoute",
        "upload_csv": "Chaje CSV",
        "upload_audio": "Chaje odyo",
        "record_voice": "Anrejistre",
        "save_voice": "Sove",
        "play": "Jwe",
        "delete": "Efase",
        "edit": "Modifye",
        "update": "Mete ajou",
        "chat_interface_label": "Chat"
    }
}

# ---------- SESSION STATE ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "embedding_model" not in st.session_state:
    with st.spinner("Loading AI model... (first time only)"):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []
if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()
if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = load_dictionaries()
if "training_access" not in st.session_state:
    st.session_state.training_access = False
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "play_audio" not in st.session_state:
    st.session_state.play_audio = None

VOICE_CACHE = load_voice_cache()

# ---------- PRE‑DEFINED VOICE MAPPING ----------
PREDEFINED_VOICES = {
    "site konbyen let ki genhen nan alfabe kreyol la": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording%20(1).wav",
    "konbyen let ki genhen nan alfabe kreyol la": "https://raw.githubusercontent.com/Deslandes1/Gesner-AI/main/recording%20(3).wav",
    "kijan ou rele": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording%20(4).wav"
}

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def get_predefined_voice_url(user_question):
    norm_q = normalize_text(user_question)
    for key, url in PREDEFINED_VOICES.items():
        if key in norm_q or norm_q.startswith(key):
            return url
    return None

# ---------- HELPER FUNCTIONS (unchanged) ----------
def save_all():
    save_training_data()
    save_dictionaries()
    save_voice_cache()

def build_tfidf():
    if st.session_state.texts:
        st.session_state.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)

def rebuild_index():
    if st.session_state.training_data:
        st.session_state.texts = [item["text"] for item in st.session_state.training_data]
        embeddings = [np.array(item["embedding"], dtype=np.float32) for item in st.session_state.training_data]
        dim = len(embeddings[0])
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.index.add(np.array(embeddings))
        build_tfidf()
    else:
        st.session_state.index = None
        st.session_state.texts = []
        st.session_state.tfidf_vectorizer = None
        st.session_state.tfidf_matrix = None

def add_to_training(text):
    if not text.strip():
        return False
    embedding = st.session_state.embedding_model.encode([text])[0]
    st.session_state.training_data.append({"text": text, "embedding": embedding.tolist()})
    rebuild_index()
    save_training_data()
    return True

def update_training_item(idx, new_text):
    if not new_text.strip():
        return False
    embedding = st.session_state.embedding_model.encode([new_text])[0]
    st.session_state.training_data[idx] = {"text": new_text, "embedding": embedding.tolist()}
    rebuild_index()
    save_training_data()
    return True

def delete_training_item(idx):
    st.session_state.training_data.pop(idx)
    rebuild_index()
    save_training_data()

def get_voice_filename(text):
    norm = text.strip().lower()
    h = hashlib.md5(norm.encode()).hexdigest()
    return h

def save_voice_for_text(text, audio_bytes):
    global VOICE_CACHE
    key = get_voice_filename(text)
    VOICE_CACHE[key] = audio_bytes
    save_voice_cache()

def get_voice_for_text(text):
    if not text:
        return None
    key = get_voice_filename(text)
    return VOICE_CACHE.get(key)

def character_picker(key_prefix, label="Insert Kreyòl characters:"):
    chars = ["e","è","E","È","o","ò","O","Ò","an","An","AN","en","En","EN","on","On","ON","oun","Oun","OUN"]
    st.markdown(f"**{label}**")
    cols = st.columns(len(chars))
    for i, ch in enumerate(chars):
        with cols[i]:
            if st.button(ch, key=f"char_{key_prefix}_{ch}"):
                if key_prefix.startswith("edit_"):
                    idx = key_prefix.split("_")[1]
                    key = f"edit_text_{idx}"
                    current = st.session_state.get(key, "")
                    st.session_state[key] = current + ch
                st.rerun()

def retrieve_facts_hybrid(query, k=5):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    query_embedding = st.session_state.embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
    distances, indices = st.session_state.index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(st.session_state.texts) and distances[0][i] < 1.2:
            results.append(st.session_state.texts[idx])
    if st.session_state.tfidf_vectorizer is not None and st.session_state.tfidf_matrix is not None:
        q_vec = st.session_state.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-k:][::-1]
        for idx in top_indices:
            if scores[idx] > 0.1 and st.session_state.texts[idx] not in results:
                results.append(st.session_state.texts[idx])
    return results[:k]

def direct_keyword_answer(query):
    q_lower = query.lower().strip()
    if "ti malice" in q_lower:
        if "kiyès" in q_lower or "who" in q_lower or "kreyatè" in q_lower:
            return "Ti Malice se yon lojisyèl edikatif ki fèt pa Gesner Deslandes pou anseye Kreyòl Ayisyen atravè jwèt ak istwa."
        if "chapit" in q_lower or "chapter" in q_lower:
            return "Ti Malice gen 12 chapit. Chapit 1: Alfabè, Chapit 2: Nonm, Chapit 3: Koulè, Chapit 4: Fanmi, Chapit 5: Manje, Chapit 6: Bèt, Chapit 7: Vèb, Chapit 8: Tan, Chapit 9: Fraz senp, Chapit 10: Konvèsasyon, Chapit 11: Pwovèb, Chapit 12: Istwa."
        if "telechaje" in q_lower or "download" in q_lower:
            return "Ou ka telechaje Ti Malice sou sitwèb globalinternet.py."
        return "Ti Malice se yon lojisyèl k ap anseye Kreyòl Ayisyen. Li gen 12 chapit ak egzèsis. Pou plis enfòmasyon, mande m 'chapit Ti Malice' oswa 'telechaje Ti Malice'."
    if any(w in q_lower for w in ["beginner", "debutan", "debutant", "aprann kreyòl deba"]):
        return "Kou Kreyòl pou debitan (Beginner): Alfabè 32 lèt, pwonon (mwen, ou, li, nou, yo), vèb 'se' ak 'gen', salitasyon (Bonjou, Bonswa), nonm 1-10, koulè debaz. Kisa ou ta renmen aprann an premye?"
    if any(w in q_lower for w in ["intermediate", "entèmedyè", "mwayen", "intermédiaire"]):
        return "Kou Kreyòl entèmedyè: Tan pase ak 'te', tan kap vini ak 'ap' oswa 'pral', nègasyon ak 'pa', pwopozisyon (nan, sou, anba), fraz konplèks ak 'ki', 'kote', 'poukisa'. Vle w pran yon egzèsis?"
    if any(w in q_lower for w in ["advanced", "avanse", "avancé"]):
        return "Kou Kreyòl avansé: Pawòl konpoze, pwovèb popilè (Dèyè mòn gen mòn, Men anpil chay pa lou), tan ki konpoze (Mwen te ap manje), vwa pasif, sijonktif (Fòk ou vini), literati kreyòl, ak analiz powèm. Eksplore youn nan sijè sa yo."
    return None

def reason_about_question(query):
    q = query.lower().strip()
    math_match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", q)
    if math_match:
        try:
            a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
            if op == '+': res = a + b
            elif op == '-': res = a - b
            elif op == '*': res = a * b
            elif op == '/': res = a / b
            else: res = None
            if res is not None:
                if isinstance(res, float) and res.is_integer():
                    res = int(res)
                return f"Repons lan se {res}."
        except: pass
    if "kapital" in q or "capital" in q:
        capitals = {"france":"Paris","ayiti":"Pòtoprens","haiti":"Port‑au‑Prince","etazini":"Washington, D.C.","usa":"Washington, D.C.","kanada":"Ottawa","brezil":"Brasília","alman":"Bèlen","itali":"Wòm","espay":"Madrid","angle":"Londr","japon":"Tokiyo"}
        for country, cap in capitals.items():
            if country in q:
                return f"Kapital {country.title()} se {cap}."
    if "ki lè li ye" in q or "what time" in q:
        now = datetime.now().strftime("%H:%M")
        return f"Kounye a li {now}."
    return None

def reason_answer(query, retrieved_facts):
    if not retrieved_facts:
        return None
    if len(retrieved_facts) == 1:
        return retrieved_facts[0]
    q_lower = query.lower()
    if any(w in q_lower for w in ["beginner", "debutan", "debutant"]):
        beginner_facts = [f for f in retrieved_facts if "beginner" in f.lower() or "debitan" in f.lower() or "alfabè" in f.lower() or "pwonon" in f.lower()]
        if beginner_facts:
            return ". ".join(beginner_facts[:3])
    if any(w in q_lower for w in ["intermediate", "entèmedyè"]):
        inter_facts = [f for f in retrieved_facts if "intermediate" in f.lower() or "entèmedyè" in f.lower() or "tan pase" in f.lower()]
        if inter_facts:
            return ". ".join(inter_facts[:3])
    if any(w in q_lower for w in ["advanced", "avanse"]):
        adv_facts = [f for f in retrieved_facts if "advanced" in f.lower() or "avanse" in f.lower() or "pwovèb" in f.lower()]
        if adv_facts:
            return ". ".join(adv_facts[:3])
    if "ti malice" in q_lower:
        malice_facts = [f for f in retrieved_facts if "ti malice" in f.lower()]
        if malice_facts:
            return ". ".join(malice_facts[:3])
    if any(word in q_lower for word in ["raconte", "rakonte", "istwa", "history", "histoire"]):
        history_facts = [f for f in retrieved_facts if any(kw in f.lower() for kw in ["endepandan", "revolisyon", "duvalier", "tranblemanntè", "1804", "1915", "1957", "bwa kayiman"])]
        if history_facts:
            combined = ". ".join(history_facts[:3])
            return combined + "."
        else:
            return retrieved_facts[0]
    return retrieved_facts[0]

def generate_response(user_input):
    normalized = user_input.strip().lower()
    if "site konbyen let ki genhen nan alfabe kreyol la" in normalized:
        answer = "A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z"
        return answer, False, False
    if re.search(r"konbyen let ki (genhen|gehen) nan alfabe kreyol la", normalized):
        answer = "Nan alfabe kreyol la genhen 32 let."
        return answer, False, False
    if "kijan ou rele" in normalized or "ki jan ou rele" in normalized:
        answer = "Non pa mwen se Gesner L'AI kreyate mwen se Gesner Deslandes nan Globalinternet.py."
        return answer, False, False

    with st.spinner("🧠 Gesner AI ap reflechi... (thinking...)"):
        time.sleep(0.8)
        direct = direct_keyword_answer(user_input)
        if direct:
            return direct, False, False
        math_result = reason_about_question(user_input)
        if math_result and ("+" in user_input or "-" in user_input or "*" in user_input or "/" in user_input):
            return math_result, False, False
        facts = retrieve_facts_hybrid(user_input, k=5)
        if facts:
            reasoned = reason_answer(user_input, facts)
            return reasoned, False, False
        logic = reason_about_question(user_input)
        if logic:
            return logic, False, False
    return "Mwen poko konn sa. Tanpri anseye m nan Sant Fòmasyon.", True, False

# ---------- AUDIO PLAYBACK ----------
def show_audio_button(text, user_question, key_suffix):
    url = get_predefined_voice_url(user_question) if user_question else None
    if url:
        if st.button("🔊", key=f"audio_btn_{key_suffix}", help="Play audio"):
            st.session_state.play_audio = ("url", url)
            st.rerun()
        return
    audio_bytes = get_voice_for_text(text)
    if audio_bytes:
        if st.button("🔊", key=f"audio_btn_{key_suffix}", help="Play audio"):
            st.session_state.play_audio = ("bytes", audio_bytes, "audio/wav")
            st.rerun()
        return

def render_audio_player():
    if st.session_state.play_audio:
        audio_type = st.session_state.play_audio[0]
        if audio_type == "url":
            url = st.session_state.play_audio[1]
            st.audio(url, format="audio/wav")
        elif audio_type == "bytes":
            _, data, mime = st.session_state.play_audio
            st.audio(data, format=mime)
        st.session_state.play_audio = None

# ---------- UI COMPONENTS (unchanged) ----------
def dictionary_manager(t):
    st.subheader(t['dictionary'])
    lang = st.selectbox("Select language", list(LANGUAGES.keys()), key="dict_lang")
    lang_code = LANGUAGES[lang]
    word = st.text_input("Word / Phrase", key="dict_word")
    meaning = st.text_area("Meaning / Translation", key="dict_meaning")
    if st.button("Add / Update", key="dict_add"):
        if word and meaning:
            st.session_state.dictionaries[lang_code][word] = meaning
            save_dictionaries()
            st.success("Saved!")
            st.rerun()
    st.markdown("---")
    st.write("**Existing entries**")
    for w, m in st.session_state.dictionaries[lang_code].items():
        col1, col2 = st.columns([3,1])
        with col1:
            st.write(f"**{w}**: {m}")
        with col2:
            if st.button("Delete", key=f"del_{lang_code}_{w}"):
                del st.session_state.dictionaries[lang_code][w]
                save_dictionaries()
                st.rerun()

def voice_training(t):
    st.subheader(t['voice_training'])
    fact_text = st.text_area(t['fact_text'], key="voice_fact")
    uploaded_audio = st.file_uploader(t['upload_audio'], type=["wav", "mp3"], key="voice_upload")
    if uploaded_audio:
        audio_bytes = uploaded_audio.read()
        st.audio(audio_bytes, format="audio/wav")
        if st.button(t['save_voice'], key="save_voice_btn"):
            save_voice_for_text(fact_text, audio_bytes)
            st.success("Voice saved!")
    st.markdown("---")
    st.write("**Existing voice mappings**")
    for idx, item in enumerate(st.session_state.training_data):
        text = item["text"]
        if get_voice_for_text(text):
            col1, col2 = st.columns([3,1])
            with col1:
                st.write(text[:60] + "..." if len(text) > 60 else text)
            with col2:
                if st.button(t['play'], key=f"play_voice_{idx}"):
                    audio_bytes = get_voice_for_text(text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")

def bulk_training(t):
    st.subheader(t['bulk_training'])
    uploaded_file = st.file_uploader(t['upload_csv'], type=["csv"], key="bulk_csv")
    if uploaded_file:
        df = csv.DictReader(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        facts = [row.get("fact") or row.get("text") for row in df]
        if facts:
            if st.button("Import facts", key="bulk_import"):
                count = 0
                for fact in facts:
                    if fact and fact.strip():
                        if add_to_training(fact.strip()):
                            count += 1
                st.success(f"Imported {count} facts.")
                st.rerun()

def manage_trained_facts(t):
    st.subheader(t['manage_facts'])
    for idx, item in enumerate(st.session_state.training_data):
        col1, col2, col3 = st.columns([4,1,1])
        with col1:
            if f"edit_{idx}" in st.session_state and st.session_state[f"edit_{idx}"]:
                new_text = st.text_area("Edit", value=item["text"], key=f"edit_text_{idx}")
                if st.button("Save", key=f"save_edit_{idx}"):
                    update_training_item(idx, new_text)
                    st.session_state[f"edit_{idx}"] = False
                    st.rerun()
            else:
                st.write(item["text"])
        with col2:
            if st.button(t['edit'], key=f"edit_btn_{idx}"):
                st.session_state[f"edit_{idx}"] = True
                st.rerun()
        with col3:
            if st.button(t['delete'], key=f"del_btn_{idx}"):
                delete_training_item(idx)
                st.rerun()

def test_training_section(t):
    st.subheader(t['test_training'])
    query = st.text_input("Test query", key="test_query")
    if st.button("Test", key="test_btn"):
        if query:
            facts = retrieve_facts_hybrid(query, k=3)
            if facts:
                st.write("**Retrieved facts:**")
                for f in facts:
                    st.write(f"- {f}")
            else:
                st.write("No relevant facts found.")

def training_center(t):
    st.markdown(f"## {t['training_center']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {t['train_new']}")
        new_fact = st.text_area(t['fact_text'], key="new_fact")
        if st.button(t['add_fact'], key="add_fact_btn"):
            if new_fact.strip():
                add_to_training(new_fact.strip())
                st.success("Fact added!")
                st.rerun()
    with col2:
        bulk_training(t)
    manage_trained_facts(t)
    test_training_section(t)

def chat_interface(t):
    st.markdown(f"<h1 style='text-align:center; color:#ffd966;'>{t['app_title']}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Mwen reponn sèlman an Kreyòl. Poze m kesyon sou alfabè, gramè, istwa Ayiti, oswa nenpòt bagay ou te anseye m.</p>", unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.conversation_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                col_text, col_btn = st.columns([10, 1])
                with col_text:
                    st.markdown(f'<div class="assistant-message" style="padding:0.5rem; border-radius:20px;">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                with col_btn:
                    if not msg.get("skip_audio", False):
                        user_q = st.session_state.conversation_history[idx-1]["content"] if idx > 0 else ""
                        show_audio_button(msg["content"], user_q, f"chat_{idx}")
            st.markdown("")
    render_audio_player()
    user_input = st.text_input(t['chat_input'], key="chat_input")
    if st.button(t['send'], use_container_width=True, key="send_btn"):
        if user_input.strip():
            answer, is_fallback, skip_audio = generate_response(user_input)
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": answer,
                "is_fallback": is_fallback,
                "skip_audio": skip_audio
            })
            st.rerun()
    if st.button(t['clear'], use_container_width=True, key="clear_btn"):
        st.session_state.conversation_history = []
        st.rerun()

def show_sidebar():
    with st.sidebar:
        st.markdown('<div class="spinning-brain">🧠</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-info">
                <strong>Gesner AI</strong><br>
                Created by <strong>Gesner Deslandes</strong><br>
                Founder of <strong>GlobalInternet.py</strong><br>
                ✉️ <a href="mailto:deslandes78@gmail.com">deslandes78@gmail.com</a><br>
                📞 +509 4738-5663<br>
                🌐 <a href="https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/" target="_blank">globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        lang_choice = st.selectbox("🌐 Interface Language", list(LANGUAGES.keys()), key="lang_select")
        st.session_state.ui_language = LANGUAGES[lang_choice]
        t = TEXTS.get(st.session_state.ui_language, TEXTS["en"])
        menu = st.radio("Menu", [t['chat_interface_label'], t['dictionary'], t['voice_training'], t['training_center']])
        return menu, t

def main():
    if not st.session_state.training_data:
        initialize_default_training()
    menu, t = show_sidebar()
    if menu == t.get('chat_interface_label', "Chat"):
        chat_interface(t)
    elif menu == t['dictionary']:
        dictionary_manager(t)
    elif menu == t['voice_training']:
        voice_training(t)
    elif menu == t['training_center']:
        training_center(t)

if __name__ == "__main__":
    main()
