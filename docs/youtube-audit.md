# Заявка на аудит YouTube API: що обмежує нас зараз і готові відповіді

## Що саме обмежує нас без аудиту

Документація `videos.insert` каже дослівно:

> All videos uploaded via the `videos.insert` endpoint from unverified API projects created
> after 28 July 2020 will be restricted to private viewing mode.

Наш проєкт `yt-factory-510411` створено 2 жовтня 2026, тобто він під це правило підпадає.
Наслідок для нас один, але головний: **відео, завантажене програмою, може назавжди лишитись
приватним, хоч би що ми просили**, і запланована публікація на ньому тихо не спрацює.
Помилки при цьому не буде — відповідь API виглядає нормальною.

Другий, менш болючий наслідок: без аудиту не можна просити більшу квоту за межі стандартних
10 000 одиниць на добу. Нам цього поки вистачає з запасом: одне завантаження коштує 1 одиницю
з окремого кошика на 100 завантажень, обкладинка ≈50 одиниць із загальних 10 000.

**Чого ми ще не знаємо.** Перше відео (`gRePZohxtuM`) завантажилось, і YouTube **прийняв**
`publishAt` на 7 жовтня 21:00 за Києвом, повернувши цю дату у відповіді. Тобто planування
формально не відхилено. Чи стане відео справді публічним того вечора — покаже автоматична
перевірка (`scripts/check_published.py`, щовечора). Якщо стане, обмеження нас не торкається
і аудит потрібен лише заради майбутньої квоти. Якщо ні — аудит стає обов'язковим. Подавати
заявку варто в обох випадках: вона безкоштовна, а розгляд триває тижнями.

## Форма

**https://support.google.com/youtube/contact/yt_api_form**

Точного переліку полів я не бачив: форма відкривається лише під акаунтом і я її не заповнював.
Нижче — відповіді на питання, які ця форма ставить за документацією і за описами розробників.
**Перед надсиланням звірте формулювання з тим, що справді запитано на екрані**, і скажіть мені,
якщо якесь поле відрізняється — перепишу.

---

## Готові відповіді

**Project number / Project ID**
`yt-factory-510411` (номер проєкту видно в Google Cloud Console на головній сторінці проєкту;
якщо форма просить саме числовий Project number, візьміть його звідти)

**OAuth client ID**
`147757322186-hj1csktsf05hq0jj4g43frlhmis9r1sr.apps.googleusercontent.com`

**API client name / Application name**
yt-factory

**Website or repository URL**
https://github.com/beautyapp800-ctrl/yt-factory-daily

**Which YouTube API Services does your project use?**
YouTube Data API v3, two endpoints only: `videos.insert` and `thumbnails.set`.

**Which OAuth scopes do you request?**
One: `https://www.googleapis.com/auth/youtube.upload`. We deliberately do not request the
broader `youtube` scope, because the project never needs to read, edit or delete anything on
the channel.

**Describe your use case / what does your application do?**

> yt-factory is a private, single-channel publishing tool that I built and run for my own
> YouTube channel. It is not a product, it has no users other than me, and it is not offered
> to anyone else.
>
> Once a day it assembles one long-form video essay on Stoic philosophy and publishes it to my
> own channel: it writes a script with a language model, narrates it with a text-to-speech
> voice, generates illustrative artwork, renders the video, and uploads it with a title,
> description and tags. Every video is uploaded to one channel — my own — and scheduled with
> `status.publishAt` so it appears at a consistent hour.
>
> The entire process runs unattended on GitHub Actions. There is no user interface, no sign-up,
> no third party, and no other person's account is ever touched.

**Who are the users of your application? How do they authenticate?**

> There is exactly one user: me, the channel owner. I authorised the application once in a
> browser, and the resulting refresh token is stored as an encrypted GitHub Actions secret used
> only by my own repository. No other person can or does sign in. The application has no login
> screen, because it has no other users.

**Is your application publicly available?**

> The source code is public on GitHub so the process is auditable, but the application itself
> is not offered to anyone. It cannot be used by a third party without their own Google Cloud
> project, their own credentials and their own channel.

**Do you handle AI-generated or synthetic content?**

> Yes, and it is declared. Every upload sets `status.containsSyntheticMedia = true`, because the
> narration is synthetic speech and the artwork is generated. Every upload also sets
> `status.selfDeclaredMadeForKids = false`. Both flags are constants in the code, not settings:
> they cannot be switched off by a configuration change, and the value YouTube returns is
> compared with what was requested and logged if they differ.

**How do you comply with the YouTube API Services Terms of Service?**

> - Only two endpoints are called, `videos.insert` and `thumbnails.set`, with a single upload
>   scope. Nothing is read, modified or deleted on the channel.
> - Nothing is scraped. No YouTube data is stored beyond the video ids of my own uploads and
>   their scheduled publication times.
> - No YouTube data is passed to anyone, sold, or used for advertising or model training.
> - Uploads go only to the channel that authorised the application, at most one video a day.
> - Synthetic content is declared on every upload, as above.
> - All content is original: scripts are newly written, the artwork is generated, and the
>   background music is licensed CC0 with each track's source recorded in the repository.
> - The application does not simulate a viewer, does not generate views, likes, subscriptions
>   or comments, and has no access to anyone else's account.

**Do you have a privacy policy?**

> Yes: https://beautyapp800-ctrl.github.io/yt-factory-daily/privacy.html
>
> It states what is true of this application: it has one user, its owner; it collects nothing
> from anyone else, because it has no interface through which anything could be submitted; and
> the token it holds carries a single scope that can upload to one channel and cannot read,
> edit or delete anything.

**How much quota do you need?**

> The default is enough. One upload a day is 1 unit from the uploads bucket, plus roughly 50
> units for the thumbnail. I am not asking for an increase; I am asking for the compliance
> review so that videos uploaded by this project are not restricted to private.

**Demo video / screenshots**

> The application has no user interface to film. What can be shown instead: the GitHub Actions
> run log for a complete daily run, end to end, and the resulting video on the channel. A run
> log is at https://github.com/beautyapp800-ctrl/yt-factory-daily/actions

---

## Що зробити перед надсиланням

1. **Перевірте номер проєкту.** У формі може стоять саме числовий *Project number*, а не
   `yt-factory-510411`. Він у Google Cloud Console на головній сторінці проєкту.
2. **Політика приватності вже є:**
   https://beautyapp800-ctrl.github.io/yt-factory-daily/privacy.html
   (вихідний текст — `docs/privacy.md`, сторінка роздається з GitHub Pages).
3. **Надсилаєте ви.** Форма прив'язується до акаунта, під яким її відкрито: відкривайте під тим
   самим, де канал M1 Stories і проєкт Cloud.
4. Відповідь приходить листом, зазвичай за кілька тижнів. Доти публікуйте вручну, якщо перевірка
   7 жовтня покаже, що автоматична публікація не спрацювала.

## Чого я в цій заявці не писав

Я не стверджував, що в нас є застосунок із інтерфейсом чи демонстраційне відео — нічого з
цього немає. Політика приватності тепер є і каже лише те, що правда. Якщо аудитор наполягатиме, це доведеться
зробити окремо, і я скажу як.
