---
title: yt-factory
---

# yt-factory

**yt-factory uses YouTube API Services.**

It is a private, automated publishing tool. Its owner runs it for one YouTube channel of their
own: each day it writes a script on Stoic philosophy, narrates it with synthetic speech,
generates the artwork, renders a long-form video and uploads it to that single channel.

It is not a product and not offered to anyone else. There is no sign-up, no login and no user
interface. The source code is public so that what the software does can be checked.

## Policies

- **[Privacy Policy](privacy.html)** — what this application collects, and from whom
- **[Terms of Use](terms.html)** — who may use it, and on what terms

## Required YouTube API Services disclosures

- [YouTube Terms of Service](https://www.youtube.com/t/terms)
- [Google Privacy Policy](https://policies.google.com/privacy) — how Google handles data
- [Revoke this application's access to your Google account](https://myaccount.google.com/permissions)
  — the owner can do this at any time, and it stops the application immediately

## How it uses the API

Two endpoints of the YouTube Data API v3, `videos.insert` and `thumbnails.set`, with one OAuth
scope: `https://www.googleapis.com/auth/youtube.upload`. That scope can upload a video and set
its thumbnail on the channel that authorised it. It cannot read, edit or delete anything on the
channel, and gives no access to any other Google service.

No YouTube data is sold, transferred, used for advertising or used to train any model. The
application never simulates a viewer and never generates views, likes, subscriptions or
comments.

Every upload declares `status.containsSyntheticMedia = true`, because the narration is
synthetic speech and the artwork is generated, and `status.selfDeclaredMadeForKids = false`.

## Source

[github.com/beautyapp800-ctrl/yt-factory-daily](https://github.com/beautyapp800-ctrl/yt-factory-daily)
