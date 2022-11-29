from datasets import load_lookups
import torch
from rich.console import Console
from constants import MAX_LENGTH
import click
from get_model import get_model
import torch.nn as nn
from typing import List, Dict, Any
from datasets import load_code_descriptions


class Infer:
    def __init__(self, model: nn.Module, dicts: List[Dict[str, Any]]):
        """

        Parameters
        ----------
        model : nn.Module
            The convolutional attention model trained on MIMIC III
            Look at the model code for the inputs that need to be sent to the model
            This is absolutely shoddy in the way that it is written. There is no way that you can
            just pass the input without the target in the original code to get the predictions
        dicts: list[dict[str, Any]]
            A list of dictionaries that the CAML code forms
            see datasets.load_lookups() function for more information
        """
        self.model = model
        self.dicts = dicts
        self.console = Console()
        self.max_length = MAX_LENGTH

        try:
            self.ind2w, self.w2ind, self.ind2c, self.c2ind, self.dv_dict = (
                self.dicts["ind2w"],
                self.dicts["w2ind"],
                self.dicts["ind2c"],
                self.dicts["c2ind"],
                self.dicts["dv"]
            )
        except KeyError:
            self.console.print("[red] Invalid dicts passed. Look at the CAML code and pass the right ones")

        self.num_labels = len(self.dicts["ind2c"])
        self.sigmoid = nn.Sigmoid()
        self.code_descriptions = load_code_descriptions()

    def predict(self, note_text: str, k: int = 10):
        """

        Parameters
        ----------
        note_text : str
            The `note_text` on which the predictions are made.
        k: int
            The topk predictions for the given text

        Returns
        -------
        list[str]
            A list of class names that are predicted by the model
            The top-k predicted by the model are returned

        """

        # Split the text into tokens
        # OOV words are given a unique index at end of vocab lookup
        text = [int(self.w2ind[w]) if w in self.w2ind else len(self.w2ind) + 1 for w in note_text.split()]
        # truncate long documents
        if len(text) > self.max_length:
            text = text[: self.max_length]

        if len(text) < self.max_length:
            text.extend([0] * (self.max_length - len(text)))

        text = torch.LongTensor(text)

        # a single text has batch size of 1
        # add the 0th dimension
        text = text.unsqueeze(0)

        # ignore the other outputs.
        # consider the first one
        logits, _, _ = self.model(text, target=None)
        logits = self.sigmoid(logits)

        # flatten the logits
        logits = logits.flatten()

        logits_sorted, indices = torch.sort(logits, descending=True)
        indices = indices.tolist()

        predicted_codes = [self.ind2c[idx] for idx in indices[:k]]
        predicted_descriptions = [self.code_descriptions[predicted_code] for predicted_code in predicted_codes]

        return predicted_codes, predicted_descriptions


@click.command()
@click.option("--label_space", type=str, required=True, help="Size of label space")
@click.option("--embed-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
              required=True,
              help="path to a file holding pre-trained embeddings")
@click.option("--filter-size", type=int, required=True, help="The size of the kernel used for convolution")
@click.option("--num-filter-maps", type=int, required=True, help="The number of output filter maps in the convolution")
@click.option("--lmbda", type=float, required=False, default=0,
              help="Used when DR CAML is used. The lambda value is used for regularization of the embeddings")
@click.option("--gpu/--no-gpu", default=False, help="Optional flag to use GPU if available")
@click.option("--public-model/--no-public-model", default=True,
              help="Optional flag for testing pre-trained models from the public github")
@click.option("--vocab", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), required=True,
              help="Path to file holding vocab word list for discretizing words")
@click.option("--version", type=str, default="mimic3", required=False, help="MIMIC3/MIMIC2 versions")
@click.option("--model_name", type=str, default="conv_attn", required=False, help="Model type.")
@click.option("--data-path", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), required=True,
              help="Path to a file containing sorted train data. dev/test splits assumed to have same name format with "
                   "'train' replaced by 'dev' and 'test'")
@click.option("--model-path", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), required=True,
              help="Path where the model parameters are stored")
def main(
        label_space,
        embed_file,
        filter_size,
        num_filter_maps,
        lmbda,
        gpu,
        public_model,
        vocab,
        version,
        model_name,
        data_path,
        model_path
):
    model = get_model(
        model_path,
        label_space,
        embed_file,
        filter_size,
        num_filter_maps,
        lmbda,
        gpu,
        public_model,
        vocab,
        version,
        model_name,
        data_path
    )
    args = dict(
        public_model=public_model,
        version=version,
        vocab=vocab,
        model=model_name,
        Y=label_space,
        data_path=data_path
    )
    dicts = load_lookups(args)

    infer = Infer(model=model, dicts=dicts)
    predicted_labels, predicted_descriptions = infer.predict(
        "admission date discharge date date of birth sex f service medicine allergies iv contrast attending first name3 lf chief complaint abdominal pain major surgical or invasive procedure exlap sigmoid colectomy w hartmann s procedure picc placement diagnostic paracentesis transesophageal echocardiogram cardioversion history of present illness 75f admitted to hospital1 location un for multilobar pneumonia with multiple previous admissions with similar comlaint and managed on w ceftriaxone and flagyl now w worsening abdominal pain since this am the patient was originally admitted last week for her pneumonia at which point her only abdominal complaint was diarrhea she was empirically started on flagyl for this and discharged on friday returning saturday with continued fevers chills the patient had had persistent fevers and chills and was planned for a ct a p this am this morning the patient had acute onset severe abdominal pain worsening throughout the day she has continued to have some diarrhea she had a previous lactate of and has not had significant recent weight loss her last colonoscopy was years ago with diverticulosis but no other major findings the ct scan obtained showed free air in the abdomen with a possible perforation in the transverse colon she was subsequently transferred to hospital1 for further management past medical history past medical history dmii htn hyperlipidemia cll cad chf cri baseline cr gout ibs lung carcinoid past surgical history right upper lobectomy appendectomy cholecystectomy cabgx5 social history lives at home with husband denies tobacco etoh illicits family history non contributory physical exam admission physical exam vitals 2l gen a o nad heent no scleral icterus mucus membranes moist cv rrr pulm crackles throughout l abd soft distension tender throughout guarding dre guaic ext no le edema le warm and well perfused discharge physical exam vitals 1l on ra gen a o x3 nad lungs cta bilaterally bibasilar crackles l r cv rrr normal s1 s2 grade ii vi holosystolic murmur best heard at lusb abd midline surgical scar open and packed with guaze with no erythema or drainage ostomy in place filled with gas and stool hypoactive bowel sounds soft no tenderness to palpation gu no exudate or erythema in inguinal folds vagina foley in place ext wwp dp pt pulses bilaterally dependent edema pertinent results labs on admission wbc rbc hgb hct mcv mch mchc rdw plt ct neuts bands lymphs monos eos baso atyps metas myelos nrbc hypochr anisocy poiklo macrocy normal microcy normal polychr normal ovalocy schisto occasional tear dr last name stitle acantho pt ptt inr pt fibrino glucose urean creat na k cl hco3 angap alt ast ck cpk alkphos amylase totbili lipase ck mb ctropnt albumin calcium phos mg art po2 pco2 ph caltco2 base xs lactate microbiology blood culture x2 no growth urine culture yeast urine culture no growth pleural fluid gram stain final no polymorphonuclear leukocytes seen no microorganisms seen fluid culture final no growth anaerobic culture final no growth mrsa screen final positive for methicillin resistant staph aureus imaging ct torso chest comparison is made with the previous study done left basilar atelectasis or consolidation and small pleural effusions are again demonstrated additional small patchy pulmonary opacities are improved enlarged right hilar and mediastinal lymph nodes are probably unchanged some are difficult to assess due to lack of intravenous contrast widely scattered atherosclerotic calcification including coronary artery calcification is again demonstrated abdomen and pelvis there is interval development of free air with a moderate amount of gas beneath the right hemidiaphragm and scattered small pockets of gas in the mesentery the patient is status post cholecystectomy there is a small amount of ascites the colonic wall appears thick although the colon is not distended which may account for this finding there is no focal hepatic lesion the pancreas adrenal glands and left kidney are unremarkable the spleen is enlarged measuring cm in long axis there is an inhomogeneous mass like density off of the posterior aspect of the right kidney this measures approximately cm x cm the right renal vein is enlarged there are prominent retroperitoneal and mesenteric lymph nodes as well as clearly enlarged left iliac lymph nodes pelvic structures are unremarkable there are degenerative changes in the spine the patient is status post median sternotomy no suspicious osteolytic or osteoblastic lesion is identified impression pneumoperitoneum in the absence of a history of recent surgery or invasive procedure this suggests bowel perforation clinical correlation is recommended there appears to be some thickening of the colonic wall although this month only be due to underdistention is there clinical evidence for colitis mild ascites splenomegaly right renal mass suspicious for tumor enlargement of the right renal vein month only be due to tumor thrombus retroperitoneal and left iliac lymphadenopathy right hilar and mediastinal lymphadenopathy small pleural effusions bilateral subsegmental atelectasis and possible pulmonary consolidation consistent with pneumonia renal ultrasound right renal mass consistent with renal cell carcinoma with tumor thrombus in the right renal vein and ivc as was seen on the torso ct of location un of no hydronephrosis is seen bilaterally limited doppler examination due to the patient s body habitus there are abnormal bilateral arterial waveforms of uncertain significance as there is absent antegrade flow in diastole seen throughout it is unclear whether this is due to technical factors preserved gross arterial and venous flow is seen within each kidney bilateral lower extremity venous dopplers no dvt in both lower extremities transthoracic echocardiogram the left atrium is elongated there is moderate symmetric left ventricular hypertrophy the left ventricular cavity is unusually small due to suboptimal technical quality a focal wall motion abnormality cannot be fully excluded overall left ventricular systolic function is normal lvef right ventricular chamber size is normal the aortic valve leaflets are mildly thickened but aortic stenosis is not present no aortic regurgitation is seen the mitral valve leaflets are mildly thickened there is no mitral valve prolapse there is moderate thickening of the mitral valve chordae no mitral regurgitation is seen there is at least mild pulmonary artery systolic hypertension there is no pericardial effusion impression suboptimal image quality probably normal right ventricular cavity size right ventricular function appears to be preserved but is difficult to assess given poor acoustic windows moderate symmetric left ventricular hypertrophy with preserved global systolic funcction probable mass in ivc at cm from junction with right atrium at least mild pulmonary hypertension cxr right picc line ends at mid svc there is no pneumothorax mild to moderate right pleural effusion and minimal left pleural effusion associated with lower lung atelectasis are unchanged and hilar contours are stable heart size is top normal unchanged no new lung opacities concerning for pneumonia impression new right picc line terminates at mid svc no pneumothorax mild to moderate right and minimal left pleural effusions are unchanged cxr rotated positioning allowing for this heart size is borderline with left ventricular configuration the lungs are hyperinflated suggesting background copd there is patchy opacity at the left base and a small left effusion raising concern for a left base pneumonic infiltrate doubt chf the right lung is grossly clear possible minimal blunting of the right costophrenic angle impression small left effusion and increased opacity over the left lower lobe suspicious for a left lower lobe infiltrate labs on discharge wbc hgb hct plts na k cl co2 bun cr gluc agap ca mg p studies on discharge none brief hospital course yo f with coronary artery disease s p cabg x type ii diabetes mellitus chronic kidney disease chronic diastolic heart failure initially admitted to hospital1 location un with recurrent multilobar pneumonia patient was transferred to hospital1 and course was complicated by acute diverticulitis with sigmoid perforation requring sigmoidectomy and hartmann s pouch acute on chronic diastolic heart failure acute on chronic renal failure and supraventricular tachycardia supraventricular tachycardia patient was tachycardic following procedure this was initially treated with home carvedilol and prn iv lopressor patient was evaluated for pulmonary embolism with lenis negative for dvt and tte negative for right heart strain a cta was deferred in setting of acute on chronic renal failure tachycardia persisted and was noted to be supraventricular tachycardia patient was given bolus doses of adenosine and labetolol without improvement she was loaded with digoxin 1mg on on cardiology was consulted who recommended discontinuing digoxin loading with amiodarone 400mg daily and electrical cardioversion patient was cardioverted on following cardioversion patient was in sinus rhythm with frequent episodes of supraventricular tachycardia in addition she was also having episodes of bradycardia down to 30s which appeared to be both sinus and junctional escape carvedilol was discontinued on due to concern for heart block she was asymptomatic during both tachy and bradycardic episodes and blood pressures remained stable she denied chest pain bradycardic episodes resolved spontaneously and patient was restarted on metoprolol 25mg po bid supraventricular tachycardia resolved and patient was discharged on metoprolol 25mg tid and amiodarone 200mg daily ekg prior to discharge confirmed sinus rhythm with frequent pvcs the patient will follow up in cardiology clinic following discharge diverticulitis c b perforation requring sigmoidectomy and hartmann s pouch patient recovered well from initial procedure and was tolerating diet at the time of discharge with good stool flatus output into ostomy laparotomy scar did not fully close and had pus requiring irrigation per surgery no further antibiotics were started the wound appeared clean without further drainage or surrounding erythema a wound vac was placed to quicken secondary intent closure patient should follow up with general surgery in weeks acute on chronic renal failure patient s baseline creatinine is and peaked at it gradually improved stable at with good urine output renal mass patient had an incidental finding of a renal mass with concern for extension into the svc urology was consulted and suggested better investigation with mri which was deferred in the setting of acute on chronic renal failure patient determined that she did not want further investigation of the mass she and her husband discussed this at length with the palliative care team and continued to agree for no further investigation hypoxia pneumonia patient was hypoxic despite euvolemia and was felt to have pneumonia she was treated with vancomycin and zosyn to be continued for days until via picc line she was on 1l at the time of discharge acute on chronic diastolic heart failure this occurred following surgery and was treated with aggressive diuresis with lasix drip she had a transudative pleural effusion s p thoracentesis with improvement in oxygenation leukocytosis peaked at infectious work up was negative with negative blood urine and pleural fluid cultures patient has a history of recurrent multilobar pneumonias for which she was initially admitted to the hospital and treated with vanc zosyn she also had three weeks of diarrhea prior to admission treated empirically with flagyl patient was treated with vanc cipro flagyl during this admission with course completed on patient had a new lll opacity on cxr and was started on vancomycin and zosyn on to treat health care associated pneumonia for days goals of care patient was followed by palliative care during admission patient was dnr dni throughout admission she elected to not have further intervention on renal mass as above no further changes were made to her goals of care transitional issues vancomycin zosyn through please check vancomycin trough prior to pm dose today f u with general surgery on f u with cardiology on patient had a new finding of renal mass but refused further investigation intervention medications on admission allopurinol 100mg daily calcitriol 5mcg po daily carvedilol 5mg po bid clonazepam 5mg po daily clonidine 1mg po bid felodipine 20mg po daily lasix 60mg po bid levemir 60u qpm sitagliptin 50mg po daily sertraline 50mg po daily renogel 800mg po tid zocor 20mg po daily baby aspirin tramadol 25mg po qid prn pain colace 100mg po daily discharge medications dabigatran etexilate mg capsule sig one capsule po bid times a day clonidine mg tablet sig one tablet po bid times a day metoprolol tartrate mg tablet sig one tablet po bid times a day amiodarone mg tablet sig one tablet po daily daily for weeks ondansetron hcl pf mg ml solution sig one injection q8h every hours as needed for nausea vancomycin in d5w gram ml piggyback sig one intravenous q 24h every hours for days day treat for days allopurinol mg tablet sig one tablet po daily daily amlodipine mg tablet sig two tablet po daily daily miconazole nitrate cream sig one appl vaginal hs at bedtime for days day oxycodone mg tablet sig one tablet po q4h every hours as needed for pain sertraline mg tablet sig one tablet po daily daily cepacol sore throat mg lozenge sig one mucous membrane prn as needed for sore throat acetaminophen mg tablet sig two tablet po q6h every hours as needed for pain miconazole nitrate powder sig one appl topical tid times a day as needed for rash insulin lispro unit ml solution sig per sliding scale subcutaneous qachs piperacillin tazobactam gram recon soln sig one intravenous q6h every hours for days day treat for days clonazepam mg tablet sig one tablet po tid times a day as needed for anxiety agitation furosemide mg tablet sig one tablet po bid times a day wound vac please place wound vac on midline laparotomy incision at pressure of please change vac q3days discharge disposition extended care facility hospital3 northeast location un discharge diagnosis primary diagnosis perforated sigmoid bowel s p sigmoidectomy with hartmann s pouch supraventricular tachycardia acute on chronic renal failure health care associated pneumonia renal mass secondary diagnosis coronary artery disease hypertension diastolic cardiomyopathy discharge condition mental status confused sometimes level of consciousness lethargic but arousable activity status out of bed with assistance to chair or wheelchair discharge instructions dear mrs known lastname it was a pleasure taking care of you during your recent admission you were admitted to hospital1 with a perforation in your bowel requiring surgery you recovered very well from the abdominal surgery you will need to have a wound vac in place for several days to help the scar heal faster in addition you had a fast heart rate which required cardioversion by the cardiologists but medications helped control your heart rate you also had a pneumonia which developed at the end of your hospitalization for which you will be getting a total of days of antibiotics medication changes start amiodarone for better heart rate control change carvedilol to metoprolol for better heart rate control change felodipine to amlodipine for blood pressure control start vancomycin through for pneumonia start zosyn through for pneumonia start dabigatran for prevention of blood clots followup instructions department general surgery hospital unit name when tuesday at pm with acute care clinic with dr first name8 namepattern2 name stitle phone telephone fax building lm hospital ward name bldg last name namepattern1 location un campus west best parking hospital ward name garage department cardiac services when tuesday at am with doctor first name name8 md m d telephone fax building sc hospital ward name clinical ctr location un campus east best parking hospital ward name garage"
    )

    print(f"Number of predicted labels {len(predicted_labels)}")
    for label, description in zip(predicted_labels, predicted_descriptions):
        print(label, description)


if __name__ == "__main__":
    main()
