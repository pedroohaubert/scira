// https://env.t3.gg/docs/nextjs#create-your-schema
import { createEnv } from '@t3-oss/env-nextjs'
import { z } from 'zod'

export const serverEnv = createEnv({
  server: {
    OPENAI_API_KEY: z.string().min(1),
    TAVILY_API_KEY: z.string().min(1),
    OPENROUTER_API_KEY: z.string().min(1),
    FIRECRAWL_API_KEY: z.string().min(1),
    EXA_API_KEY: z.string().min(1),
    YT_ENDPOINT: z.string().min(1),
  },
  experimental__runtimeEnv: process.env,
})
