---
title: Terms of Use
---

# Terms of Use — yt-factory

**Last updated: 4 October 2026**

## 1. What this software is

yt-factory is a private, automated publishing tool. Its owner runs it for one YouTube channel
of their own. It produces a long-form video on Stoic philosophy each day — writing the script,
narrating it with synthetic speech, generating the artwork, rendering the video — and uploads
it to that single channel.

It is not a product, not a hosted service, and not offered to anyone. There is no sign-up, no
login, no user interface and no customer. The source code is published so that what the
software does can be inspected; publishing the source is not an offer of a service.

## 2. Who may use it

**One person: the owner of the channel it publishes to.** Nobody else has access, and there is
no mechanism by which anyone else could obtain access. The running instance is authorised by a
single OAuth token held by the owner, in the owner's own Google Cloud project, and it can
upload only to the channel that granted it.

Another person could in principle copy the public source code and run their own separate
instance. If they do, it is their deployment, with their own Google Cloud project, their own
credentials and their own channel. Nothing in these terms gives them a service, and the owner
of this instance provides them with nothing.

## 3. No service, no warranty, no liability to third parties

Because no service is offered to anyone, there is nothing for a third party to rely on. The
software is provided as-is, without warranty of any kind, express or implied. The owner accepts
no obligation and no liability to any third party arising from this software or its source
code, to the fullest extent permitted by law.

The owner remains fully responsible for the content published on their own channel and for
their own compliance with YouTube's policies.

## 4. YouTube API Services

This application uses YouTube API Services. It calls exactly two endpoints of the YouTube Data
API v3 — `videos.insert` and `thumbnails.set` — with a single OAuth scope,
`https://www.googleapis.com/auth/youtube.upload`. It cannot read, edit or delete anything on
the channel, and it has no access to any other Google service.

By using this application the owner agrees to be bound by the
[YouTube Terms of Service](https://www.youtube.com/t/terms). Google's handling of data is
described in the [Google Privacy Policy](https://policies.google.com/privacy). The owner may
revoke this application's access to their Google account at any time at
[https://myaccount.google.com/permissions](https://myaccount.google.com/permissions); doing so
stops the application immediately and permanently.

The application does not and will not: sell or transfer YouTube data to anyone; use YouTube
data for advertising or to train any model; simulate a viewer; generate views, likes,
subscriptions or comments; access any account other than the one that authorised it; or
circumvent any part of the YouTube service.

## 5. Automated and synthetic content

Videos are produced automatically. The narration is synthetic speech and the artwork is
generated. Every upload declares this to YouTube with `status.containsSyntheticMedia = true`,
and declares `status.selfDeclaredMadeForKids = false`. Both are constants in the source, not
configuration: they cannot be switched off by changing a setting, and the values YouTube
returns are compared with the values requested and recorded when they differ.

Background music is licensed CC0, with the source of each track recorded in the repository.

## 6. Privacy

See the [Privacy Policy](privacy.html). In short: the application collects nothing from anyone
other than its owner, because it has no interface through which anything could be submitted.

## 7. Changes

Any change to these terms is a commit in the public repository, so the full history can be read
at [github.com/beautyapp800-ctrl/yt-factory-daily](https://github.com/beautyapp800-ctrl/yt-factory-daily).

## 8. Contact

Through the repository's issue tracker:
[github.com/beautyapp800-ctrl/yt-factory-daily/issues](https://github.com/beautyapp800-ctrl/yt-factory-daily/issues)
