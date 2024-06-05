# Guide du LLM

## PARTIE II. Développements autour des LLMs (pour les data scientists)

### I. Revue technique de l’état de l’art LLM (Malo Jérôme)


#### 1. Principe pré-entraînement / fine-tuning

https://www.entrypointai.com/blog/pre-training-vs-fine-tuning-vs-in-context-learning-of-large-language-models/ 

#### 2. Architectures principales LLM

##### A. L'architecture Transformer

- [Papier original **'Attention Is All You Need'**](https://arxiv.org/abs/1706.03762)
- [Explication illustrée et très détaillée](http://jalammar.github.io/illustrated-transformer/)

##### B. Encoder-only, encoder-decoder, decoder-only

Les LLMs basés sur des architectures Transformers appartiennent à l’une des 3 catégories suivantes : 

- **Modèle « encoder-only »** : Ils sont basés uniquement sur la partie décodeur des Transformers. Leur pré-entraînement est souvent basé sur la reconstruction de phrases : à chaque étape, le modèle a accès à une phrase entière, sauf certains mots qui ont été masqués, et apprend à retrouver ces mots masqués. Ces modèles sont adaptés pour des tâches de classification, de reconnaissance d’entités nommées (NER), de réponses aux questions, etc. Ils ont aujourd’hui perdu en popularité, mais leurs représentants les plus connus (BERT, RoBERTa, DistilBERT, CamemBERT, etc.) sont encore très utilisés, et restent un choix intéressant selon la tâche, grâce à leur compréhension fine du langage et à leur petite taille.

- **Modèle « decoder-only »** : Ils sont basés uniquement sur la partie décodeur des Transformers. Ces modèles sont aujourd’hui la norme, et l’immense majorité des LLMs actuels utilisent cette architecture. Leur pré-entraînement est basé sur la prédiction du prochain token : à chaque étape, le modèle a accès au début d’une phrase, et apprend à prédire le token suivant. Pour cette raison, ces modèles sont également qualifiés d’« autorégressifs ». Les modèles GPT (2, 3, 4), Llama (2, 3), Mistral, Gemini, etc. sont tous des decoder-only.

- **Modèle « encoder-decoder »** : Ils utilisent les deux blocs des Transformers. 

https://medium.com/artificial-corner/discovering-llm-structures-decoder-only-encoder-only-or-decoder-encoder-5036b0e9e88 

##### C. Mixture of Experts (MoE)

Explication détaillée des MoE (exemple de Mixtral) : https://huggingface.co/blog/moe 

##### D. Nouvelles architectures : Mamba, Jamba, etc.

Le principal inconvénient architectural des Transformers est leur complexité quadratique par rapport à la taille de l'entrée (qui vient du calcul quadratique de l'attention). **Mamba** est une architecture récente (Décembre 2023) qui s'affranchit du mécanisme d'attention, au profit de briques SSM (Structured State Space Models). L'intérêt principal de cette architecture est sa complexité linéaire par rapport à la taille de l'entrée.

**Jamba** est une nouvelle architecture hybride, à mi-chemin entre le Transformer et Mamba. Cela semble permettre un niveau de performance élevé, une gestion des contextes très longs, un temps d'inférence nettement plus court, et des exigences mémoires bien moindres.

Liens des papiers originaux : 
- [Mamba](https://arxiv.org/abs/2312.00752) 
- [Jamba](https://arxiv.org/abs/2403.19887) 

#### 3. Méthodes de fine-tuning

##### A. Fine-tuning supervisé

###### a. Fine-tuning complet

- [Implémentation HuggingFace](https://huggingface.co/docs/transformers/training)

###### b. Fine-tuning efficace (PEFT) : LoRA, QLoRA, DoRA, etc.

PEFT = Parameter-Efficient Fine-Tuning | LoRA = Low-Rank Adaptation | QLoRA = Quantized Low-Rank Adaptation | DoRA = Weight-Decomposed Low-Rank Adaptation

Ré-entraîner entièrement un LLM est très coûteux en termes d'infrastructure et de données, et n'est donc pas à la portée de n'importe quelle organisation. Des méthodes « efficaces » ont été créées pour rendre le fine-tuning facilement accessible, dont la plus connue et la plus populaire est LoRA (pour Low-Rank Adaptation). Son fonctionnement repose sur deux éléments : 

- **L'adaptation** : Les poids du modèle pré-entraîné sont gelés pendant l'entraînement. Ce sont des poids supplémentaires (ceux de l'adapteur) qui vont être entraînés. Cela permet de garder l'entièreté du modèle pré-entraîné tel quel, et de rajouter uniquement la partie spécifique à chaque tâche. Entre autres, il est ainsi possible, avec un seul modèle de base, d'héberger plusieurs modèles spécialisés à moindre coût. Le papier [LoRA Land](https://arxiv.org/abs/2405.00732) explique d'ailleurs comment faire tenir 25 versions de Mistral 7B fine-tunés avec LoRA sur un seul GPU A100.  

- **Le rang faible** : Les poids additionnels peuvent être choisis de beaucoup de manières. Avec LoRA, certaines couches du modèle (les couches d'attention ou les couches linéaires par exemple) sont sélectionnées, et les poids de ces couches sont exprimés comme une multiplication de deux matrices de rangs faibles, ce qui réduit grandement le nombre de poids à entraîner (la valeur de ce rang étant un hyperparamètre de l'entraînement). En fonction de la valeur de ce rang et des couches sélectionnées, il est ainsi possible d'entraîner uniquement 1 ou 2 % du nombre de paramètres global du modèle pré-entraîné, sans que cela n'affecte trop les performances du fine-tuning.

D'autres approches de PEFT (Parameter-Efficient Fine-Tuning) ont vu le jour, dont la plupart s'inspirent de LoRA. Parmi les plus connues, QLoRA permet d'appliquer LoRA sur des modèles quantifiés, et DoRA propose un raffinement de l'adapteur de LoRA. 

- [Guide théorique très clair sur le PEFT (principe, avantages, etc.) avec un focus sur LoRA](https://www.leewayhertz.com/parameter-efficient-fine-tuning/)
- [Guide pratique / Implémentation HugginFace](https://huggingface.co/blog/gemma-peft)

Liens des papiers originaux : 
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314) 
- [DoRA](https://arxiv.org/abs/2402.09353) 

##### B. RLHF et RLAIF

RLHF = Reinforcement Learning from Human Feedback | RLAIF = Reinforcement Learning from Artificial Intelligence Feedback

- [Introduction au RLHF](https://huggingface.co/blog/rlhf)

###### a. PPO

PPO = Proximal Policy Optimization

- [Explication théorique](https://huggingface.co/blog/deep-rl-ppo)
- [Implémentation HuggingFace](https://huggingface.co/docs/trl/main/en/ppo_trainer)

https://medium.com/@oleglatypov/a-comprehensive-guide-to-proximal-policy-optimization-ppo-in-ai-82edab5db200 

###### b. DPO, KTO

DPO = Direct Preference Optimization | KTO = Kahneman-Tversky Optimization

- [Explication théorique](https://huggingface.co/blog/pref-tuning)
- [Guide pratique / Implémentation HugginFace](https://huggingface.co/blog/dpo-trl) 

Liens des papiers originaux : 
- [DPO](https://arxiv.org/abs/2305.18290) 
- [KTO](https://arxiv.org/abs/2402.01306) 

##### C. Fine-tuning d'embeddings

*Plutôt dans la partie RAG ?*

##### D. Divers

###### a. Prompt-tuning

- [Lien du papier](https://arxiv.org/abs/2104.08691)

###### b. ReFT et LoReFT

ReFT = Representation Fine-Tuning | LoReFT = Low-Rank Linear Subspace ReFT

- [Lien du papier](https://arxiv.org/abs/2404.03592)

#### 4. Prompt engineering

##### A. Bonnes pratiques

Il faut avant tout garder à l'esprit que le prompt engineering est une discipline très empirique, qui demande beaucoup d'itérations pour obtenir le meilleur prompt par rapport au résultat souhaité. Bien qu'il n'existe pas de méthode systématique et efficace pour optimiser un prompt, certaines pratiques sont devenues la norme. Par exemple, voici quelques bonnes pratiques : 
- **Donner un rôle au modèle** : Par exemple, dire au modèle qu'il est un magistrat honnête et impartial pourra l'aider à générer du texte formel, neutre et juridique. Le rôle est bien sûr à adapter en fonction des exigences de chaque tâche.
- **Structurer le prompt** : Il est important de bien différencier le *prompt système* du *prompt utilisateur*. Le premier donnera des instructions générales quant au style, à la tâche, au contexte, etc., alors que le second pourra donner des instructions spécifiques ou un texte à analyser. Il est également pertinent d'organiser ou de séparer clairement les instructions.
- **Etre le plus précis possible** : 
- **Contraindre le modèle au maximum** : 
- **Donner des exemples** : Cf. paragraphe suivant.


##### B. 0-shot, 1-shot, few-shot prompting

La façon la plus intuitive d'adresser une requête à un LLM est de formuler des instructions les plus précises possibles. Ce faisant, on espère que le modèle comprendra ces instructions et répondra en conséquence. Pour des tâches nouvelles, auxquelles le modèle n'a pas nécessairement été confronté durant son (pré)-entraînement, on appelle cette méthode du 0-shot prompting : le modèle n'a pas de référence ou d'exemple de réponse attendue.

Pour pallier ce manque de référence, il est possible (et, en fonction de la tâche, souvent recommandé) d'ajouter des exemples de paires entrée/sortie dans le prompt que l'on adresse au modèle : cela donne du 1-shot (un exemple) ou du few-shot (plusieurs exemples) prompting. Plus les exemples sont proches de la requête initiale, plus le modèle saura précisément comment répondre. Cela permet ainsi au modèle de s'adapter, à moindre coût, à une tâche très spécifique ou particulière.

- [Guide pratique (avec exemples)](https://www.prompthub.us/blog/the-few-shot-prompting-guide)
 
##### C. Chain of Thought (CoT) reasoning

Sur certaines tâches qui demandent un raisonnement (par exemple la résolution d'un problème mathématique simple), les LLM naturellement ne sont pas très bons. Pour augmenter leurs capacités de raisonnement, une stratégie classique consiste à leur demander de raisonner et de réfléchir étape par étape. 

Les modèles les plus récents ayant nettement progressé en raisonnement, il est possible qu'ils raisonnent naturellement étape par étape sur des questions simples. Pour des questions ou des raisonnements plus complexes, il sera cependant probablement plus efficace de proposer une logique de raisonnement au modèle, en explicitant les différentes étapes. 

Il est également possible de combiner le CoT reasoning avec du few-shot prompting, *i.e.* de donner des exemples de raisonnement étape par étape au modèle. 

- [Guide détaillé](https://www.mercity.ai/blog-post/guide-to-chain-of-thought-prompting)

##### D. RAG

RAG = Retrieval Augmented Generation

Le principe est de rajouter du contexte dans le prompt du LLM, pour lui donner accès à des données spécifiques et pertinentes. Cf. partie sur la RAG.

##### E. Reverse prompt engineering ?



#### 5. Quoi faire quand ?

##### A. Utiliser un LLM

La première question à se poser est la nécessité ou non d’utiliser un LLM. Certaines tâches peuvent se résoudre avec un LLM, mais ce n’est pas toujours la solution la plus pertinente. Par exemple, un LLM est normalement capable de parser un fichier xml sans problème, mais un script naïf sera largement aussi efficace, à bien moindre coût (environnemental, humain, financier). L’utilisation d’un LLM doit venir d’un besoin de compréhension fine du langage naturel. 

**Donner quelques exemples de cas d'usages**

##### B. Quel(s) modèle(s) utiliser

Beaucoup d’éléments sont à prendre en compte lors du choix du modèle à utiliser. Parmi les plus importants : 

- **Sa taille** : Exprimée généralement en milliards (B) de paramètres (Llama-3 8B possède 8 milliards de paramètres, Mistral 7B en possède 7 milliards, etc.), elle influe fortement sur les performances du modèles et les exigences techniques. Un « petit » LLM de 8 milliards de paramètres pourra tourner sur un GPU modeste avec une VRAM de 32 GB (voire moins si l’on utilise un modèle quantifié, cf. …), tandis qu’un LLM de taille moyenne de 70 milliards de paramètres nécessitera 2 GPU puissants avec 80 GB de VRAM.

- **Son multilinguisme** : La plupart des modèles sont entraînés sur une immense majorité de données anglaises (plus de 90 % pour Llama-2, contre moins de 0,1 % de données françaises). Les modèles incluant plus de français (Mistral ?) dans leurs données d’entraînement sont naturellement plus efficaces sur du français. 

- **Son temps d’inférence** : Généralement directement lié à la taille du modèle, certaines architectures (MoE) permettent cependant d’avoir un temps d’inférence plus court.

- **Ses performances générales** : Beaucoup de benchmarks publics évaluent les LLMs sur des tâches généralistes et variées. Un bon point de départ est de regarder [le Leaderboard](https://chat.lmsys.org/?leaderboard) qui recense la plupart des modèles connus. 

- **Ses performances spécifiques** : Les benchmarks généralistes ne sont pas forcément pertinents pour certains cas d’usages, car ils ne sont pas spécifiques à la tâche, aux données, etc. Il peut être intéressant de développer un pipeline d’évaluation spécifique (cf…).

##### C. Quand faire du prompt engineering

Si vous êtes dans l'un des cas suivants, le prompt engineering peut être une bonne option : 

- Pas beaucoup de ressources disponibles
- Besoin d'un outil laissé à la disposition des utilisateurs, avec une grande liberté
- Les réponses requises sont très formattées ou très spécifiques

##### D. Quand faire de la RAG

Si vous êtes dans l'un des cas suivants, la RAG peut être une bonne option : 

- Besoin de réponses à jour, régulièrement et facilement actualisées
- Besoin de sourcer les réponses ou de diminuer les hallucinations
- Besoin d'enrichir les réponses avec des données spécifiques 
- Besoin d'une application qui ne dépend pas d'un modèle spécifique (généralisabilité), et dont les utilisateurs ne connaissent pas l'IA générative

##### E. Quand faire du fine-tuning

Si vous êtes dans l'un des cas suivants, le fine-tuning peut être une bonne option : 

- Besoin d'une terminologie ou d'un style spécifique 
- Besoin d'enrichir les réponses avec des données spécifiques
- Ressources (GPU, data scientists) disponibles 
- Données disponibles en quantité et qualité suffisantes
- Besoin d'une application qui ne dépend pas d'un modèle spécifique (généralisabilité), et dont les utilisateurs ne connaissent pas l'IA générative

##### F. Combiner plusieurs techniques

RAG + fine-tuning = [RAFT](https://arxiv.org/abs/2403.10131)